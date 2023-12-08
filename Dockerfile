FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y libgles2-mesa-dev libglib2.0-0 swig libgl1-mesa-glx libosmesa6-dev libglfw3 libglew2.1

# ENV MUJOCO_GL=osmesa
# ENV PYOPENGL_PLATFORM=osmesa
# ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

RUN pip install --upgrade pip
COPY ./requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt