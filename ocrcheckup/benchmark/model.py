import traceback
import numpy as np

class OCRBaseModel():
  name = "name"
  version = "version"

  is_cloud = False
  is_lmm = False

  def __init__(self):
    return None
  
  def test(self):
    try:
        img = np.zeros([100,100,3],dtype=np.uint8)
        img.fill(255)

        self.evaluate(img)
    except:
        print("This model you just initated is not working correctly")
        traceback.print_exc()
        return False
    
    return True

  def evaluate(self,image):
    # Should return either a string or a object with: `result` (str) and `cost` (float) (in US dollars)
    raise NotImplementedError("Evaluate function must be implemented by a subclass")

  def run_for_eval(self,image):
    start_time = time.perf_counter()
    result = self.evaluate(image)
    elapsed_time = time.perf_counter()-start_time

    if type(result) is not str:
      result = result["result"]
      cost = result["cost"]

    return {
        "start_time": start_time,
        "elapsed_time": elapsed_time,
        "result": result,
        "cost": cost or None
    }
