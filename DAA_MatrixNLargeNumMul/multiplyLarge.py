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
	print ''.join(str(x) for x in product)

def largeMultiply(a,b):
	if a==0 and b==0:
		return
	if (len(a)/2):
		a.insert(0,0)
	elif(len(b)/2):
		b.insert(0,0)
	
def main(argv):
	l_size = 0
	t = []
	with open(argv[1],'r') as f:
		lines = f.readlines()
	for i in lines:
		t.append(list(i.rstrip()))
	multiply(t[0], t[1])

if __name__=="__main__":
	main(sys.argv)
