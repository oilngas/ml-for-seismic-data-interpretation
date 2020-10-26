import os
import numpy as np
import pandas as pd
import raster_geometry as rg
import s3fs
import argparse
import process_instance_by_inline as inline_processing
import process_instance_by_crossline as crossline_processing

from tqdm import tqdm

## Data extraction methods

def extract_raw_annotations(file_path, start_inline, start_xline, z_step):
    """
    Opens ASCII-annotations file stored on S3 bucket and 
    transforms to a suitable format by extracting the information
    and converting string values to numerical values.
    """
    fs = s3fs.S3FileSystem()
    
    print("Reading annotations file...")
    with fs.open(file_path) as annotations_file:
        # Data in format
        # [fold_name, x, y, z, inline_id, point_inline_id]
        fold_dat = [i.strip().split() for i in (l.decode('utf-8') for l in annotations_file.readlines())]
        
        for fold in fold_dat:
            fold[1] = int(fold[1]) - start_inline
            fold[2] = int(fold[2]) - start_xline
            fold[3] = round(float(fold[3])/z_step)
            fold[4] = int(fold[4])
            fold[5] = int(fold[5])
                
        return fold_dat
    
    
def split_annotations_to_fold_instance_dfs(annotations):
    """
    Creates a pandas DataFrame from the extracted
    annotations and divides it fault-instance-wise.
    
    Return a list of data frames for corresponding
    fault instances.
    """
    
    fold_df = pd.DataFrame(annotations)
    fold_df.columns = ['instance', 'inline', 'crossline', 'z', 'stick', 'node']
    
    # Group by folds and split to separate data frames
    fold_split_dfs = [x for _, x in fold_df.groupby('instance')]
    
    return fold_split_dfs

def process_cube(fold_split_dfs: list, 
                 volume_shape: tuple, 
                 start_inline: int, 
                 start_xline: int, 
                 z_step: int,
                 num_points_per_stick = 50):
    """
    Creates an empty cube for a single fold instance
    and fills the values of the cube with the corresponding points.
    
    Additionally could draw lines between the points in the matrix.
    """
    
    masks = np.zeros((volume_shape[0], volume_shape[1], volume_shape[2]), dtype = np.int32)
    df_columnts = ['instance', 'inline', 'crossline', 'z', 'stick', 'node']
    labels_df = pd.DataFrame(columns = df_columnts)
    class_label = 1
    
    # Iterate over fold splits
    for fault_instance in tqdm(fold_split_dfs):
        instance = fault_instance.iloc[0]['instance']
        
        instance_label_direction = instance.split("_")[-1]
        
        if instance_label_direction == "i":
            print("Processing instance {} by inlines".format(instance))
            mask, instance_df = inline_processing.process_instance_inl(fault_instance,
                                                                       volume_shape,
                                                                       class_label, 
                                                                       instance, 
                                                                       start_inline, 
                                                                       start_xline, 
                                                                       z_step, 
                                                                       num_points_per_stick)
            # Add labels from this instance to general information
            masks = masks + mask
            labels_df = labels_df.append(instance_df)
            del mask
            
        elif instance_label_direction == "c":
            print("Processing instance {} by crosslines".format(instance))
            mask, instance_df = crossline_processing.process_instance_crx(fault_instance,
                                                                       volume_shape,
                                                                       class_label, 
                                                                       instance, 
                                                                       start_inline, 
                                                                       start_xline, 
                                                                       z_step, 
                                                                       num_points_per_stick)
            # Add labels from this instance to general information
            masks = masks + mask
            labels_df = labels_df.append(instance_df)
            del mask
        else:
            print("Unrecognized labels direction for instance {}".format(instance))
            
        
        class_label += 1
     
    # Process masks to be in binary format
    masks[masks > 0] = 1
    
    # Post-process labels to remove duplicates
    # and sort according to ASCII-required format
    labels_df = labels_df.drop_duplicates(subset = ['inline', 'crossline', 'z'])
    labels_df = labels_df.sort_values(by = ['instance', 'stick', 'node'])
    labels_df = labels_df[df_columnts]
    
    return masks, labels_df


def main():
    # Set random seed to ensure reproducability
    np.random.seed(42)
    
    # Parse input arguments
    parsed_args = parser.parse_args()
    
    # Extract ASCII file path
    labels_path = parsed_args.seismic_ascii_path
    
    # Extract start inline, crossline and z-step
    start_inline = parsed_args.start_inline
    start_crossline = parsed_args.start_xline
    z_step = parsed_args.z_step
    
    # Extract volume shape for labels processing
    volume_shape = tuple(parsed_args.volume_shape)
    
    # Extract output data information
    output_path = parsed_args.output_path
    output_cube_name = parsed_args.output_cube_name
    output_labels_name = parsed_args.output_labels_name
    
    # Extract annotations as DataFrame
    annotations = extract_raw_annotations(file_path = labels_path,
                                          start_inline = start_inline,
                                          start_xline = start_crossline,
                                          z_step = z_step)
    
    # Split annotation data into fault-instance wise splits
    cube_split_df = split_annotations_to_fold_instance_dfs(annotations)
    
    # Process the cube
    cube, labels = process_cube(fold_split_dfs = cube_split_df,
                                volume_shape = volume_shape,
                                start_inline = start_inline, 
                                start_xline = start_crossline, 
                                z_step = z_step)
    
    print("Saving the results of processing...")
    labels.to_csv('{}.csv'.format(output_labels_name), index = False, sep = ' ')
    #np.save('{}.npy'.format(output_cube_name), cube)
 
    
## Argument parser

parser = argparse.ArgumentParser()

parser.add_argument("--seismic_ascii_path", 
                    type = str, 
                    help="Path to ASCII labels for the seismic data")

parser.add_argument("--start_inline", 
                    type = int, 
                    help="Starting inline for instances in the cube.")

parser.add_argument("--start_xline", 
                    type = int, 
                    help="Starting crossline for instances in the cube.")

parser.add_argument("--z_step", 
                    type = int, 
                    help="Z-Step for instances in the cube. NOTE: This script works with fixed value of z-step for now.")

parser.add_argument("--volume_shape", 
                    type = int, 
                    nargs="+", 
                    help="Shape of the volume")

parser.add_argument("--output_path", 
                    type = str, 
                    help="Path to store outputs of the processing.") 

parser.add_argument("--output_cube_name", 
                    type = str, 
                    default = "cube_processed",
                    help="Name for the output cube file.")

parser.add_argument("--output_labels_name", 
                    type = str, 
                    default = "labels_processed",
                    help="Name for the labels output file.")
    
if __name__ == "__main__":
    main()