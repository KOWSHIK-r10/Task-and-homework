class Robot:
    def __init__(self, model, energy_level=100):
        self._model = model
        self.__energy_level = energy_level

    def get_energy(self):
        return self.__energy_level
    
    def set_energy(self, energy):
        try:
            energy = int(energy)  # Convert to integer
            if 0 <= energy <= 100:
                self.__energy_level = energy
            else:
                print("Invalid energy level")
        except ValueError:
            print("Energy must be a number")

    def charge(self, amount):
        try:
            amount = int(amount)  # Convert to integer
            new_energy = self.__energy_level + amount
            self.__energy_level = min(new_energy, 100)
        except ValueError:
            print("Charge amount must be a number")

robot = Robot("t1000")

try:
    print(robot.__energy_level)
except AttributeError as e:
    print(f"error: {e}")            

print(f"current energy: {robot.get_energy()}")
robot.set_energy("85")  # String converted to int
print(f"updated energy: {robot.get_energy()}")

robot.set_energy("129")  # Out of range

robot.charge("15")  # String converted to int
print(f"charged energy level: {robot.get_energy()}")