# Dockerfile for setting up a CUDA environment with Python and other dependencies

#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS dev

RUN apt-get update -y \
    && apt-get install -y python3-pip git

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-12.4/compat/

WORKDIR /workspace

# install build and runtime dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch \
    transformers \
    fastapi \
    uvicorn \
    huggingface_hub \
    pydantic

# install flash-attn
RUN pip install --no-cache-dir --no-build-isolation flash-attn

# cuda arch list used by torch
# can be useful for both `dev` and `test`
# explicitly set the list to avoid issues with torch 2.2
# see https://github.com/pytorch/pytorch/pull/123243
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
#################### BASE BUILD IMAGE ####################
#################### BASE BUILD IMAGE ####################
