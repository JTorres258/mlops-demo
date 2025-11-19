FROM tensorflow/tensorflow:2.20.0-gpu
WORKDIR /workspace

# Optional: additional dependencies
# COPY requirements.txt .
# RUN pip install -r requirements.txt

CMD ["bash"]
