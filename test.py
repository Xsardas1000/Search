__author__ = 'maksim'

from textblob import TextBlob

text = '''
Игра престолов
'''
blob = TextBlob(text)
print(blob.translate(to="en"))

b = TextBlob(u"بسيط هو أفضل من مجمع")
print(b.detect_language())

print(type(blob.translate(to="en")))
p = blob.translate(to="en")
print(type(p.string))

print("\033[44;36mHello World!\033[m \033[44;36mHello World!\033[m")

