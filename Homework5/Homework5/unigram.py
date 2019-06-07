import math   # This will import math module
import glob
import Homework2_Dataset_Generator

# max ' ' = 32, max = '~'
def unigram(text):
    list = [0] * 95
    for i, c in enumerate(text):
        index = ord(c)-32
        if index > 94 or index < 0:
            continue
        list[index] = list[index]+1
    return list


def countCharacters(text):
    count = 0
    for i, c in enumerate(text):
        index = ord(c)-32
        if index > 94 or index < 0:  # Count only characters that we use
            continue
        count = count+1
    return count


def countWords(text):
    return len(text.split())  # split whitespaces & get their length


def countSentences(text):
    return len(text.split('.'))  # split dots & get their length


def normalize(vector):
    sum = 0
    res = [0.0] * len(vector)
    for c in vector:
        sum += c*c
    dist = math.sqrt(sum)
    for i,c in enumerate(vector):
        res[i] = c/dist
    return res


ncu = open('./results_ncu.txt', 'w')
rawcu = open('./results_rawcu.txt', 'w')


avg = [0, 0, 0]  # [#Character, #Word, #Sentece]
for x in range(1000, 1025):
    for y in range(1, 5):
        file = open("./CASIS/%d_%d.txt" % (x, y), "r")
        raw = file.read()  # Read contents of the file
        file.close()  # We don't need that file anymore.
        rawResult = unigram(raw)
        #  Counters
        avg[0] += countCharacters(raw)
        avg[1] += countWords(raw)
        avg[2] += countSentences(raw)

        rawString = ','.join(str(v) for v in rawResult)
        normalizedResult = normalize(rawResult)
        normalizedString = ','.join(str(v) for v in normalizedResult)
        rawcu.write("%d_%d,%s\n" % (x, y, rawString))
        ncu.write("%d_%d,%s\n" % (x, y, normalizedString))

ncu.close()
rawcu.close()
Homework2_Dataset_Generator.Generate_TSNCU("results")


#  Statistics file
stats = open('./casis25_stats.txt', 'w')
stats.write('Character #%d\nWord #%d\nSentence #%d\n' % (avg[0]/100, avg[1]/100, avg[2]/100))
stats.close()


sncu = open('./msst_ncu.txt', 'w')
srawcu = open('./msst_rawcu.txt', 'w')

avg = [0, 0, 0]  # [#Character, #Word, #Sentece]
samples = glob.glob("./MSST/*.txt")  # Put every text file in the MSST folder into a list
for fileName in samples:
    baseName = fileName.split('\\')[-1].split('.')[0]  # Delete folder names & extension
    file = open(fileName, "r", encoding='utf-8')  # Force encoding to utf8
    raw = file.read()  # Read contents of the file
    file.close()  # We don't need that file anymore.
    #  Counters
    avg[0] += countCharacters(raw)
    avg[1] += countWords(raw)
    avg[2] += countSentences(raw)
    rawResult = unigram(raw)  # Generate unigram from text
    rawString = ','.join(str(v) for v in rawResult)
    normalizedResult = normalize(rawResult)
    normalizedString = ','.join(str(v) for v in normalizedResult)
    srawcu.write("%s,%s\n" % (baseName, rawString))
    sncu.write("%s,%s\n" % (baseName, normalizedString))

sncu.close()
srawcu.close()
Homework2_Dataset_Generator.Generate_TSNCU("msst")



attacks = open('./amask.txt', 'w')

samples = glob.glob("./amask/*.txt")  # Put every text file in the MSST folder into a list
for fileName in samples:
    baseName = fileName.split('\\')[-1].split('.')[0]  # Delete folder names & extension
    print(fileName)
    file = open(fileName, "r", encoding='utf-8', errors='ignore')  # Force encoding to utf8
    raw = file.read()  # Read contents of the file
    file.close()  # We don't need that file anymore.
    #  Counters
    rawResult = unigram(raw)  # Generate unigram from text
    rawString = ','.join(str(v) for v in rawResult)
    attacks .write("%s,%s\n" % (baseName, rawString))

attacks.close()

attacks = open('./amask2_rawcu.txt', 'w')

samples = glob.glob("./amask2/*.txt")  # Put every text file in the MSST folder into a list
for fileName in samples:
    baseName = fileName.split('\\')[-1].split('.')[0]  # Delete folder names & extension
    print(fileName)
    file = open(fileName, "r", encoding='utf-8', errors='ignore')  # Force encoding to utf8
    raw = file.read()  # Read contents of the file
    file.close()  # We don't need that file anymore.
    #  Counters
    rawResult = unigram(raw)  # Generate unigram from text
    rawString = ','.join(str(v) for v in rawResult)
    attacks .write("%s,%s\n" % (baseName, rawString))

attacks.close()

#  Statistics file
stats = open('./msst_stats.txt', 'w')
stats.write('Character #%d\nWord #%d\nSentence #%d\n' % (avg[0]/len(samples), avg[1]/len(samples), avg[2]/len(samples)))
stats.close()

