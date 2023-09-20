import time
import Bwork_voice
now_start = "Запустить сейчас"

for i in range(1):
    Bwork_voice.speek_text("Ожидаем сигнала")
    time.sleep(1)
    Bwork_voice.speek_text(now_start)
    Bwork_voice.speek_text("С этого места начнёт работать программа")



