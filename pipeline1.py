from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

labels = ['','','','','']
cb774db0d1 = classifier(" I`d have responded, if I were going")
a42 = classifier(" Sooo SAD I will miss you here in San Diego!!!")
c60f138 = classifier("my boss is bullying me...")
c003ef = classifier(" what interview! leave me alone")
bd9e861 = classifier(" Sons of ****, why couldn`t they put them on the releases we already bought")
b57f3990 = classifier("http://www.dothebouncy.com/smf - some shameless plugging for the best Rangers forum on earth")
c6d75b1 = classifier("2am feedings for the baby are fun when he is all smiles and coos")
c0bb8 = classifier("Soooo high")
e050245fbd = classifier("Both of you")
fc2cbefa9d = classifier("Journey!? Wow... u just became cooler.  hehe... (is that possible!?)")

print(cb774db0d1)
print(a42)
print(c60f138)
print(c003ef)
print(bd9e861)
print(b57f3990)
print(c6d75b1)
print(c0bb8)
print(e050245fbd)
print(fc2cbefa9d)