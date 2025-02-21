class Dog:
    def __init__(self,name,age):
        self.name = name 
        self.age = age
    
    def bark(self):
        print(self.name + " is barking " ) 

    def sleep(self):
        print(self.name + " is sleeping")

d1 = Dog("mummy",8)
d1.bark()
d1.sleep()  