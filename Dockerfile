# FROM pytorch/pytorch:latest # this is not compatible with cuda 11.4 (which is currently installed on the DGX)
# FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

WORKDIR /app

# Copy the requirements file first for better caching
COPY LLaMA_VQA/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

RUN mkdir /.cache && chown -R 61610:45000 /.cache

# Copy only the essential codebase and scripts
COPY LLaMA_VQA /app/LLaMA_VQA
COPY start_experiment.sh /app/
RUN chmod +x /app/start_experiment.sh



CMD ["/bin/bash", "start_experiment.sh"]