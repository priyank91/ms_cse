'''
number_generator.py

Property of :
	1. Priyank Arora
	2. Sandeep Satome
	3. Harshini 
	4. Kelvin Thomas

'''

import random
import sys as sys
import os as os
def generate(number):
    return int("1"+"0"*(number-1)),int("9"*number)

def main(argv):
	if len(argv) < 4:
		print('Command Usage:\n $> python number_generator.py <NUM> <filename.txt> <NumOfMultiplicants>')
		sys.exit(0)
	elif int(argv[1]) < 0:
		print('range of the number is positive (1, 10)')
		sys.exit(0)
	a, b = generate(pow(2,int(argv[1])))
	
	with open(argv[2],'w') as f:
		for x in range(int(argv[3])):
			f.write(str(random.randint(a,b))+"\n")
	#os.system('cat %s' %(argv[2]))

if __name__ == "__main__":
	main(sys.argv)

