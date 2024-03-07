import dataclasses

@dataclasses.dataclass
class PromptVariables:
  task_prompt: str = ""
  one_shot_image_path: str = ""
  one_shot_answer: str = ""
  cot_prompt: str = ""
  cot_answer: str = ""

@dataclasses.dataclass
class MensaPromptVariable(PromptVariables):
  task_prompt: str = "The image displays an intelligence test question featuring a 3x3 grid with nine boxes, one of which is empty and marked with a question mark (?). Your task is to select the correct shape from six options (labeled A to F) to fill the empty box, completing the pattern that links all the shapes together. You must first give your explanation and then provide your answer at the end of your response in the format: 'The correct answer is: _'."

  one_shot_image_path: str = "data/manually_created/images/fused/easy.jpg" #For demonstration purpose.

  one_shot_answer: str = """\
  The grid:\n1. Top left box: square with a cross sign.\n2. Top center box: circle with a star.\n3. Top right box:  Empty triangle.\n4. Middle left box: empty square. \n5. Middle center box: circle with a cross sign. \n6. Middle right box: triangle with a star. \n7. Bottom left box: square with a star. \n8. Bottom center box: empty circle. \n9. Bottom right box: ’?’.\nThe alternative shapes are:\nA. Triangle with a star .\nB. Triangle with a plus sign.\nC. Circle with a cross sign.\nD. Circle with a star.\nE. Empty triangle.\nF. Triangle with a cross sign.\
  Analyzing the grid, there seems to be a pattern involving both the shapes and the symbols inside them:\n- The symbols inside the shapes alternate between a cross, a star, and then a blank space as you move horizontally across each row.\n- The shapes themselves also alternate within each row – this is seen with the square, circle, and triangle repeating in each row in that order."\
  Answer: "F"\
  """


@dataclasses.dataclass
class ITPatternPromptVariable(PromptVariables):
  task_prompt: str = "The image displays an intelligence test question in the form of a matrix pattern puzzle. There are eight shapes provided, with one space left blank, indicated by a question mark. The shapes are organized into three rows, each containing three boxes. Your task is to select the correct shape from six options (labeled A to F) to fill the empty box, completing the pattern that links all the shapes together. You must first give your explanation and then provide your answer at the end of your response in the format: 'The correct answer is: _'."

  one_shot_image_path: str = "data/manually_created/images/fused/easy.jpg" #For demonstration purpose.

  one_shot_answer: str = """\
  The grid:\n1. Top left box: square with a cross sign.\n2. Top center box: circle with a star.\n3. Top right box:  Empty triangle.\n4. Middle left box: empty square. \n5. Middle center box: circle with a cross sign. \n6. Middle right box: triangle with a star. \n7. Bottom left box: square with a star. \n8. Bottom center box: empty circle. \n9. Bottom right box: ’?’.\nThe alternative shapes are:\nA. Triangle with a star .\nB. Triangle with a plus sign.\nC. Circle with a cross sign.\nD. Circle with a star.\nE. Empty triangle.\nF. Triangle with a cross sign.\
  Analyzing the grid, there seems to be a pattern involving both the shapes and the symbols inside them:\n- The symbols inside the shapes alternate between a cross, a star, and then a blank space as you move horizontally across each row.\n- The shapes themselves also alternate within each row – this is seen with the square, circle, and triangle repeating in each row in that order."\
  Answer: "F"\
  """

@dataclasses.dataclass
class RavenPromptVariable(PromptVariables):
  task_prompt: str = "The image displays an intelligence test question featuring a 3x3 grid with nine boxes, where the 9th box is marked with a question mark (?). Your task is to select the correct shape from eight options (labeled A to H) to fill the 9th box, completing the pattern that links all the shapes together. You must first give your explanation and then provide your answer at the end of your response in the format: 'The correct answer is: _'."
  
  one_shot_image_path: str = "data/manually_created/images/fused/easy.jpg" #For demonstration purpose.

  one_shot_answer: str = """\
  The grid:\n1. Top left box: square with a cross sign.\n2. Top center box: circle with a star.\n3. Top right box:  Empty triangle.\n4. Middle left box: empty square. \n5. Middle center box: circle with a cross sign. \n6. Middle right box: triangle with a star. \n7. Bottom left box: square with a star. \n8. Bottom center box: empty circle. \n9. Bottom right box: ’?’.\nThe alternative shapes are:\nA. Triangle with a star .\nB. Triangle with a plus sign.\nC. Circle with a cross sign.\nD. Circle with a star.\nE. Empty triangle.\nF. Triangle with a cross sign.\
  Analyzing the grid, there seems to be a pattern involving both the shapes and the symbols inside them:\n- The symbols inside the shapes alternate between a cross, a star, and then a blank space as you move horizontally across each row.\n- The shapes themselves also alternate within each row – this is seen with the square, circle, and triangle repeating in each row in that order."\
  Answer: "F"\
  """

all_prompts = {
  "mensa" : MensaPromptVariable(),
  "it-pattern": ITPatternPromptVariable(),
  "raven": RavenPromptVariable(),
  "mensa_test": MensaPromptVariable()
}
