import string
import re

class Normalizer:
    #punctuations = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    letters = string.ascii_letters
    word2digit = {
        "zero": "0",
        "uno": "1",
        "due": "2",
        "tre": "3",
        "quattro": "4",
        "cinque": "5",
        "sei": "6",
        "sette": "7",
        "otto": "8",
        "nove": "9",
        "dieci": "10",
        "undici": "11",
        "dodici": "12",
        "tredici": "13",
        "quattordici": "14",
        "quindici": "15",
        "sedici": "16",
        "diciassette": "17",
        "diciotto": "18",
        "diciannove": "19",
        "venti": "20"
    }

    def normalize_immobili(self, immobili):
        for immobile in immobili:
            for key in immobile.keys():
                immobile[key] = self.normalize(immobile[key], key)
        return immobili


    def normalize(self, s, type):
        if type.lower() == "lotto":
            return self.lotto(s)
        elif type.lower() == "foglio":
            return self.foglio(s)
        elif type.lower() == "particella":
            return self.particella(s)
        elif type.lower() == "sub":
            return self.sub(s)
        else:
            return s


    def lotto(self, s):
        if s == None:
            return "lotto unico"
        s = str(s)
        s = self.normalize_spaces(s)
        s = s.lower()

        if "unico" in s:
            return "lotto unico"

        # converti "due" in "2"
        words = s.split()
        replaced_text = ' '.join([self.word2digit.get(word, word) for word in words])
        return replaced_text


    def foglio(self, s):
        if s == None:
            return ""
        #s = self.normalize_spaces(s)
        s = str(s)
        s = s.lower()
        return s


    def particella(self, s):
        if s == None:
            return ""
        #s = self.normalize_spaces(s)
        s = str(s)
        s = s.lower()
        return s


    def sub(self, s):
        if s == None:
            return ""
        #s = self.normalize_spaces(s)
        s = str(s)
        s = s.lower()
        return s


    def normalize_spaces(selfself, s):
        if s == None:
            return ""
        return re.sub(r'\s+', ' ', s).strip()

