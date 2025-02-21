class Dog:
    def __init__(self,name,age):
        self.name = name 
        self.age = age
    
    def bark(self):
        print(self.name + " is barking ") 

    def sleep(self):
        print(self.name + " is sleeping")

class puppy(Dog):
    def play(self):
        print(self.name + " is playing")

mypuppy = puppy("kakashi",2)

mypuppy.bark()
mypuppy.sleep()
mypuppy.play()
