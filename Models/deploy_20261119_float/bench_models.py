"""Time each TFLite model on this machine with the hub's own interpreter (run from ~/birdserver)."""
import sys, time
import numpy as np
from server.bird_inference import _interpreter_class

Interpreter, runtime = _interpreter_class()
print("runtime:", runtime)
for path in sys.argv[1:]:
    for threads in (1, 4):
        it = Interpreter(model_path=path, num_threads=threads)
        it.allocate_tensors()
        inp = it.get_input_details()[0]
        x = np.random.randint(0, 255, inp["shape"]).astype(inp["dtype"])
        it.set_tensor(inp["index"], x); it.invoke()  # warm-up
        t0 = time.perf_counter()
        for _ in range(3):
            it.set_tensor(inp["index"], x); it.invoke()
        print(f"{(time.perf_counter() - t0) / 3 * 1000:9.0f} ms  threads={threads}  {np.dtype(inp['dtype']).name:7}  {path}")
