import torch
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    "pszemraj/long-t5-tglobal-base-16384-book-summary",
    device=0 if torch.cuda.is_available() else -1,
)
long_text = """În informatică, inteligența artificială (IA) este inteligența expusă de mașini, spre deosebire de inteligența naturală, expusă de oameni și de unele animale. Informatica definește cercetarea IA ca studiu al „agenților inteligenți⁠(d)”: orice dispozitiv care își percepe mediul și efectuează acțiuni care maximizează șansa de a-și atinge cu succes obiectivele.[1][2] Mai exact, Kaplan și Haenlein definesc IA ca fiind „capacitatea unui sistem de a interpreta corect datele externe, de a învăța din astfel de date și de a folosi ceea ce a învățat pentru a-și atinge obiective și sarcini specifice printr-o adaptare flexibilă”.[3] Termenul „inteligență artificială” este utilizat colocvial pentru a descrie mașinile care imită funcțiile „cognitive” pe care le asociază oamenii cu alte minți umane, cum ar fi „învățarea” și „rezolvarea problemelor”.[4][5]

Întrucât mașinile devin din ce în ce mai capabile, sarcinile considerate a necesita „inteligență” sunt deseori eliminate din definiția IA, un fenomen cunoscut sub numele de efectul IA⁠(d). O observație în teorema lui Tesler spune că „IA este ceea ce nu a fost încă făcut”.[6] De exemplu, recunoașterea optică a caracterelor este adesea exclusă din domeniul AI, după ce a devenit o tehnologie de rutină.[7] Capacitățile moderne ale mașinilor clasificate în general ca IA includ înțelegerea vorbirii umane,[8] concurarea la cel mai înalt nivel a unor sisteme de jocuri de strategie (cum ar fi șah și go),[9] autovehiculele autonome și rutarea inteligentă în rețelele de distribuție a conținutului, și simulările militare⁠(d)."""

result = summarizer(long_text, max_length = 100)
print(result[0]["summary_text"])
