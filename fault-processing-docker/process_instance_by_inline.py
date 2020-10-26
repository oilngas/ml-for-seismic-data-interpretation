import os
import numpy as np
import pandas as pd
import raster_geometry as rg

# Inline direction processing logic

def split_fault_instances_by_sticks(fold_df):
    """
    Splits a DataFrame representing single fault instance
    by inlines and returns as list of DataFrames
    
    This is useful for polyline building in corresponding
    inlines as only related points would be in the same DataFrame.
    """
    
    return[x for _, x in fold_df.groupby('stick')]

def process_instance_inl(fault_instance, 
                         volume_shape: tuple, 
                         class_label: int,
                         instance_name: str,
                         start_inline: int,
                         start_xline: int,
                         z_step: int, 
                         num_points_per_stick: int):
    
    print("Inserting label information into cube...")
    mask_points = process_fault_instances_by_inline(fault_instance, 
                                                    volume_shape, 
                                                    class_label)
    
    print("Interpolating surface from point cloud...")
    mask = connect_sticks_to_polyline_by_inline(mask_points, 
                                                class_label)
    
    print("Processing instance to ASCII format...")
    label_df = masks_to_ascii_inlines(mask_lines = mask,
                                       instance_name = instance_name, 
                                       start_inline = start_inline, 
                                       start_xline = start_xline, 
                                       z_step = z_step,
                                       num_points_per_inline = num_points_per_stick)
    
    return mask, label_df

def process_fault_instances_by_inline(fault_instance, 
                                      volume_shape: tuple, 
                                      class_label: int):
    """
    Connects the corresponding points between sticks
    to form a raw unfilled 3d-curve. Processes the instance
    by inline direction.
    """
    
    # Create empty cube for storing fault instance points
    print("Creating empty mask cube...")
    mask = np.zeros(volume_shape, dtype = np.int32)
        
    # Split fold by sticks
    sticks_split = split_fault_instances_by_sticks(fault_instance)
    
    for stick_split_index in range(len(sticks_split) - 1):
        curr_split = sticks_split[stick_split_index]
        next_split = sticks_split[stick_split_index + 1]
                
        # Sort splits by depth in both sticks
        curr_split = curr_split.sort_values('z', ascending = True)
        next_split = next_split.sort_values('z', ascending = True)
        
        # Connect the upper bounds of the inter-plane
        x_c_min, z_c_min, y_c_min = curr_split.iloc[0]['inline'], curr_split.iloc[0]['z'], curr_split.iloc[0]['crossline']
        x_n_min, z_n_min, y_n_min = next_split.iloc[0]['inline'], next_split.iloc[0]['z'], next_split.iloc[0]['crossline']
        between_points_bresenham = set(rg.bresenham_lines(((x_c_min, z_c_min, y_c_min), (x_n_min, z_n_min, y_n_min)), closed=True))
        for point in between_points_bresenham:
            mask[point[0], point[1], point[2]] = class_label
                
        # Connect the lower bounds of the inter-plane
        x_c_max, z_c_max, y_c_max = curr_split.iloc[-1]['inline'], curr_split.iloc[-1]['z'], curr_split.iloc[-1]['crossline']
        x_n_max, z_n_max, y_n_max = next_split.iloc[-1]['inline'], next_split.iloc[-1]['z'], next_split.iloc[-1]['crossline']
        between_points_bresenham = set(rg.bresenham_lines(((x_c_max, z_c_max, y_c_max), (x_n_max, z_n_max, y_n_max)), closed=True))
        for point in between_points_bresenham:
            mask[point[0], point[1], point[2]] = class_label
            
        # Connect intermediate nodes with edges (lines) with
        # a minimum weight
        # Since we may have different number of nodes left in
        # two sticks we must select minimum number from the two
        # and iterate over them
        for st_index in range(1, min(len(next_split), len(curr_split)) -1):
            x_n, z_n, y_n = next_split.iloc[st_index]['inline'], next_split.iloc[st_index]['z'], next_split.iloc[st_index]['crossline']
            x_c, z_c, y_c = curr_split.iloc[st_index]['inline'], curr_split.iloc[st_index]['z'], curr_split.iloc[st_index]['crossline']
                
            between_points_bresenham = set(rg.bresenham_lines(((x_n, z_n, y_n), (x_c, z_c, y_c)), closed=True))
            
            for point in between_points_bresenham:
                mask[point[0], point[1], point[2]] = class_label
  
        
    return mask

def connect_sticks_to_polyline_by_inline(masks_orig, class_label: int):
    """
    Iterates on the masks cube which holds the
    contour of the instance and connects the polylines
    """
    masks = masks_orig.copy()
    
    for slice_index in range(masks_orig.shape[0]):
        if masks_orig[slice_index, :, :].sum() != 0:
            z, y = masks_orig[slice_index, :, :].nonzero()

            # Sort the coordinates depth-wise using 
            # Decorate, Sort, Undecorate approach
            z, y = (list(t) for t in zip(*sorted(zip(z, y))))

            # Connect the corresponding coordinates
            for p_index in range(len(z) - 1):
                between_points_bresenham = set(rg.bresenham_lines(((slice_index, z[p_index], y[p_index]), (slice_index, z[p_index + 1], y[p_index + 1])), closed=True))

                for point in between_points_bresenham:
                    masks[point[0], point[1], point[2]] = class_label
                
    return masks


def masks_to_ascii_inlines(mask_lines,
                   instance_name: str, 
                   start_inline: int, 
                   start_xline: int, 
                   z_step: int,
                   num_points_per_inline: int):
    
    fold_df_columns = ['instance', 'inline', 'crossline', 'z', 'stick', 'node']
    fold_df = pd.DataFrame(columns = fold_df_columns)
    
    # Additionally generate a DataFrame with inline splitting
    # for this instance to load in OpenDTect
    stick_counter = 0
    
    for crossline in range(mask_lines.shape[0]):
        mask_crossline = mask_lines[crossline, :, :]
        
        # If there are points on this inline slice
        if mask_crossline.sum() != 0:
            # List of dictionaries for adding to DF
            points = []
            node_counter = 0
            
            # Extract non-zero points of the fault on this inline
            z, y = mask_crossline.nonzero()
                
            # Extract the coordinates of line start and end
            z_min, z_max = min(z), max(z)
            z_min_id, z_max_id = np.where(z == z_min)[0][0], np.where(z == z_max)[0][0]
            y_min, y_max = y[z_min_id], y[z_max_id]
            
            # Start of the line
            points.append({'instance': instance_name, 
                           'inline': crossline + start_inline,
                           'crossline': y_min + start_xline,
                           'z': z_min * z_step,
                           'stick': stick_counter,
                           'node': node_counter})
            
            node_counter += 1
            
            # Extract points along the line with fixed step
            random_indicies = np.arange(start = 0, stop = len(z), step = num_points_per_inline)
            
            z_rand = z[random_indicies]
            y_rand = y[random_indicies]
            
            # Add each point to the DataFrame
            for p_index in range(len(z_rand)):
                points.append({'instance': instance_name, 
                               'inline': crossline + start_inline,
                               'crossline': y_rand[p_index] + start_xline,
                               'z': z_rand[p_index] * z_step,
                               'stick': stick_counter,
                               'node': node_counter})
                
                node_counter += 1
                
                        
            # End of the line
            points.append({'instance': instance_name, 
                           'inline': crossline + start_inline,
                           'crossline': y_max + start_xline,
                           'z': z_max * z_step,
                           'stick': stick_counter,
                           'node': node_counter})
            
            node_counter += 1
                            
            points_df = pd.DataFrame(points)
            points_df = points_df.sort_values('z', ascending = True)
            
            # Temporary solution
            tmp_stick = 0
            for index, row in points_df.iterrows():
                points_df.at[index, 'node'] = tmp_stick
                tmp_stick += 1
            
            
            fold_df = fold_df.append(points_df)
            
            stick_counter += 1
            
    return fold_df