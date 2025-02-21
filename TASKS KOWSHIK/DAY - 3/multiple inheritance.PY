class dog():
    def bark(self):
        print( "dog is barking!")
    
    def sleep(self):
        print("dog is sleeping.")
class cat():

    def meow(self):
        print("cat sound like meow.")
    
    def purr(self):
        print("cat will purr.")

class hybrid(cat,dog):
    def traits(self):
        print("i have both cat and dog traits")

hybrid_pet = hybrid()
hybrid_pet.bark()
hybrid_pet.meow()
hybrid_pet.sleep()
hybrid_pet.purr()