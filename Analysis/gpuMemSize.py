import re

def sizeof_fmt(num):
    for x in ['bytes','KB','MB','GB']:
        if num < 1024.0:
            return "%6.1f %s" % (num, x)
        num /= 1024.0
    return "%6.1f %s" % (num, 'TB')

f = open('../InteractiveSpaceCvCore/Detector.cpp', 'r')
pStart = re.compile('^Detector::Detector\(.+\)\s*:\s*$')
pEnd = re.compile('^\{$')
pGpu = re.compile('_(\w+Gpu)\(\w+\.(\w+),\s*\w+\.(\w+),\s*(\w+)\)')

isStarted = False

dims = {
	'rgbWidth': 640,
	'rgbHeight': 480,
	'depthWidth': 640,
	'depthHeight': 480
}

sizes = {
	'CV_32F': 4,
	'CV_16U': 2,
	'CV_8UC3': 3,
	'CV_32FC3': 12,
	'CV_8U': 1,
	'CV_32SC1': 4
}

total = 0
for line in f:
	if ( (not isStarted) and pStart.match(line)):
		isStarted = True
		continue
	
	if (isStarted):
		if (pEnd.match(line)):
			break
			
		else:
			m = pGpu.search(line)
			if (m):
				#print m.group(1,2,3,4)
				varName = m.group(1)
				dim1 = dims[m.group(2)]
				dim2 = dims[m.group(3)]
				px = sizes[m.group(4)]
				bytes = dim1 * dim2 * px
				
				out = [varName.ljust(30), str(dim1), str(dim2), str(px), sizeof_fmt(bytes)]
				print '\t'.join(out)
				total = total + bytes
				
print
print 'Total'.ljust(30), '\t\t\t\t', sizeof_fmt(total)
