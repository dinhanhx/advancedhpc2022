from numba import cuda
from numba.cuda.cudadrv import enums
show = print

show(cuda.detect())
show(cuda.devices)
show(cuda.gpus)
device = cuda.select_device(0)
show(cuda.current_context().get_memory_info())
my_sms = device.MULTIPROCESSOR_COUNT
my_cc = device.compute_capability
cores_per_sm = 64
total_cores = cores_per_sm*my_sms
show(f"GPU compute capability: {my_cc}")
show(f"GPU total number of SMs: {my_sms}")
show(f"total cores: {total_cores}")
