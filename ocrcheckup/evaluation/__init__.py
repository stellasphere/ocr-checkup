import Levenshtein

class StringMetrics():
  def __init__(
      self,a,b,
      capitalization_sensitive=True,
      linebreak_sensitive=True,
      stripping=True,
      **kwargs
      ):

    self.a = a
    self.b = b
    self.capitalization_sensitive = capitalization_sensitive
    self.linebreak_sensitive = linebreak_sensitive
    self.stripping = stripping

    if not self.capitalization_sensitive:
      self.a, self.b = self.lowercase(self.a, self.b)

    if not linebreak_sensitive:
      self.a, self.b = self.remove_linebreaks(self.a,self.b)

    if stripping:
      self.a = self.a.strip()
      self.b = self.b.strip()

  @staticmethod
  def lowercase(*args):
    return tuple(string.lower() for string in args)
  @staticmethod
  def remove_linebreaks(*args):
    return tuple(' '.join(string.splitlines()) for string in args)

  EVALUATION_METHODS = ["levenshtein_ratio","correct"]

  def levenshtein_ratio(self):
    ratio = Levenshtein.ratio(self.a,self.b)
    return ratio

  def correct(self):
    correct = 1 if self.a==self.b else 0
    return correct

  def evaluate(self,methods=EVALUATION_METHODS):
    return tuple(getattr(self,method)() for method in methods)

