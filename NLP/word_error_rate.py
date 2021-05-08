import numpy

def cer(r: str, h: str):
    """
    Calculation of CER with Levenshtein distance.
    """
    r = list(r)
    h = list(h)
    # initialisation
    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint16)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i


    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(r)][len(h)] / float(len(r))

if __name__ == "__main__":
    r = '从卡耐基梅隆大学几代研发人员开始，本文对过去40年人们从语音识别技术进步所获得的启示进行了探讨。'
    h = '从卡耐基梅隆大学几代研发人员开始，对过去40年人们从ASR技术进步所获得的启示进行了深入探讨。'
    r = [x for x in r]
    h = [x for x in h]


    print(cer(r, h))