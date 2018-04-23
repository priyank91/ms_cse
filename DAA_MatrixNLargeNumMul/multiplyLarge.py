import sys as sys

def multiply(a,b):
	product = [0]*(len(a)+len(b))
	for j in xrange(len(b)-1,-1,-1):
		carry=0
		for i in xrange(len(a)-1,-1,-1):
			product[i+j+1] += carry+(int(a[i])*int(b[j]))
			carry = (product[i+j+1])/10
			product[i+j+1] %= 10
		product[j] += carry
	return ''.join(str(x) for x in product)

def SumOfList(list1, list2):
	return [str(int(x) + int(y)) for x, y in zip(list1, list2)]

def splitMultiplicants(mul, length):
	return mul[:length], mul[length:]

def largeMultiply(a,b):
	if len(a) == 1 or len(b) == 1:
		return int(multiply(a,b))		 

	m2 = (max(len(a),len(b))/2)

	A, B = splitMultiplicants(a,m2)
	C, D = splitMultiplicants(b,m2)
	
	AC = largeMultiply(A,C)
	PQ = largeMultiply(SumOfList(A,B), SumOfList(C,D))
	BD = largeMultiply(B,D)

	return (AC*pow(10,2*m2)) + (PQ-BD-AC)*pow(10,m2) + BD


def main(argv):
	l_size = 0
	t = []
	with open(argv[1],'r') as f:
		lines = f.readlines()
	for i in lines:
		t.append(list(i.rstrip()))
	print multiply(t[0], t[1])
	print largeMultiply(t[0],t[1])

if __name__=="__main__":
	main(sys.argv)
