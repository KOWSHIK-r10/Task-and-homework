class animal:
    def speak(self):
        raise NotImplementedError
class dog(animal):
    def speak(self):
        return "Woof Woof"
class cat(animal):
    def speak(self):
        return "Meow Meow"

animal_speak  =[dog(),cat()]
for animal in animal_speak:
    print(animal.speak())