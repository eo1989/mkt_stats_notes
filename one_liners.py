# %%
class Car:
	def __init__(self, speed, color):
		self.speed = speed
		self.color = color


ferrari = Car(400, 'red')
tesla = Car(220, 'white')

print(ferrari.color)
print(tesla.speed)

# %%
if tmp := get_value(): x = tmp