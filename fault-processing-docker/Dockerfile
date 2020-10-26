FROM python:3.6
WORKDIR .
COPY process_instance_by_crossline.py .
COPY process_instance_by_inline.py .
COPY process_labels_cube.py .
COPY requirements.txt .
RUN ["pip", "install", "-r", "requirements.txt"]
ENTRYPOINT ["python", "process_labels_cube.py"]