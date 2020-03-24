#!/usr/bin/env python3

import argparse
import csv

def parseLogs():
	parser = argparse.ArgumentParser()
	parser.add_argument('--files', '-f', required=True, nargs='+', type=str,
						help='The logfiles you want to parse')
	parser.add_argument('--out', '-o', required=False, type=str,
						help='The output file')
	args = parser.parse_args()
	files = args.files
	if args.out is not None:
		out = args.out
	else:
		out = "../tmp.csv"

	# [(testType,
		# [(noiseType, 
			# [(noiseValue,
				# [(qzmiType, recognitionPct)])])])]
	data = []
	cols = []
	for file in files:
		data, cols = parseFile(file, data, cols)
	printAsCsv(data, cols, out)

def parseFile(file, data, cols):
	file = open(file, 'r') 
	lines = file.readlines()
	lines = [line.strip() for line in lines]
	emptyLineCount = 0
	testType = ''
	qzmiType = ''
	noiseType = ''

	for i in range(len(lines)):
		if  emptyLineCount >= 2 and lines[i] != '':
			qzmiType, testType, noiseType = [t.split('.')[1] for t in lines[i].split(' ')]
			if qzmiType not in cols:
				cols.append(qzmiType)
		elif emptyLineCount == 1 and lines[i] != '':
			# print(qzmiType, testType, noiseType)
			if noiseType == "CLEAN":
				noiseValue = None
			elif noiseType == "GAUSS":
				noiseValue = lines[i].split(' ')[-1]
			elif noiseType == "GAUSS_NO_ROUND":
				noiseValue = lines[i].split(' ')[-1]
			elif noiseType == "SALT":
				noiseValue = lines[i].split(' ')[-1][:-1]

			result = lines[i + 1].split(' ')[0]
			
			i1 = len(data)
			for j in range(len(data)):
				if data[j][0] == testType:
					i1 = j
					break
			if i1 == len(data):
				data.append((testType, []))
			
			i2 = len(data[i1][1])
			for j in range(len(data[i1][1])):
				if data[i1][1][j][0] == noiseType:
					i2 = j
					break
			if i2 == len(data[i1][1]):
				data[i1][1].append((noiseType, []))
			
			i3 = len(data[i1][1][i2][1])
			for j in range(len(data[i1][1][i2][1])):
				if data[i1][1][i2][1][j][0] == noiseValue:
					i3 = j
					break
			if i3 == len(data[i1][1][i2][1]):
				data[i1][1][i2][1].append((noiseValue, []))
			
			data[i1][1][i2][1][i3][1].append((qzmiType, result))

		if lines[i] == '':
			emptyLineCount += 1
		else:
			emptyLineCount = 0
	return data, cols

def printAsCsv(data, cols, outFile):
	with open(outFile, mode='w') as file:
		csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

		csv_writer.writerow(['', '', '', *cols])
		for test in data:
			isFirst1 = True
			for noise in test[1]:
				isFirst2 = True
				for noiseVal in noise[1]:
					dictionary = dict(noiseVal[1])
					results = [dictionary.get(col, '') for col in cols]
					csv_writer.writerow([test[0] if isFirst1 else '', noise[0] if isFirst2 else '', noiseVal[0] if noiseVal[0] is not None else '', *results])
					isFirst2 = False
				isFirst1 = False

if __name__ == '__main__':
	parseLogs()