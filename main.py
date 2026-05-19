"""
Champions League AI Bot — бэкенд на Groq (бесплатно)

Запуск через Python:  python main.py
Запуск через uvicorn: uvicorn main:app --reload
Сайт:                 http://localhost:8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

# ── загружаем GROQ_API_KEY из файла .env ──
load_dotenv()

app = FastAPI(title="Champions League AI Bot")

# разрешаем браузеру обращаться к серверу
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# папка static → там лежит index.html (наш сайт)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── схема входящего запроса ──
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []   # история для памяти разговора


# ── личность бота ──
SYSTEM_PROMPT = """Ты — экспертный ИИ-ассистент по UEFA Champions League (Лига Чемпионов УЕФА).

Ты знаешь абсолютно всё о:
• Истории турнира с 1955 года (тогда назывался Кубок европейских чемпионов)
• Всех победителях, финалах, результатах
• Текущем формате турнира (групповой этап, плей-офф, финал)
• Клубах: Реал Мадрид, Барселона, Бавария, Ливерпуль, Манчестер Сити и всех других
• Легендарных игроках: Месси, Роналду, Зидан, Пушкаш, Ван Бастен и др.
• Рекордах и статистике: бомбардиры, победители, посещаемость
• Гимне, трофее "Большеухий Кубок", символике УЕФА
• Скандалах, драматичных матчах, чудесных камбэках

Правила общения:
- Отвечай на том языке, на котором пишет пользователь (русский, казахский, английский)
- Если вопрос НЕ про футбол или Лигу Чемпионов — вежливо откажи и предложи спросить про ЛЧ
- Отвечай живо, интересно, с конкретными цифрами и фактами
- Используй эмодзи умеренно для живости текста
- Длинные ответы разбивай на абзацы для читаемости"""


@app.get("/")
async def root():
    """При открытии http://localhost:8000 → перенаправляет на сайт."""
    return RedirectResponse(url="/static/index.html")


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Принимает вопрос и историю → отправляет в Groq → возвращает ответ.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY не найден. Проверь файл .env"
        )

    client = Groq(api_key=api_key)

    # собираем: системный промпт + история + новый вопрос
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + request.history
        + [{"role": "user", "content": request.message}]
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # лучшая бесплатная модель Groq
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
        reply = response.choices[0].message.content
        return {"reply": reply}

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "invalid_api_key" in error_msg.lower():
            raise HTTPException(status_code=401, detail="Неверный GROQ_API_KEY. Проверь ключ на console.groq.com")
        elif "rate_limit" in error_msg.lower():
            raise HTTPException(status_code=429, detail="Слишком много запросов. Подожди немного.")
        else:
            raise HTTPException(status_code=500, detail=f"Ошибка Groq: {error_msg}")


# ══════════════════════════════════════════════════
# ЭТА ЧАСТЬ ПОЗВОЛЯЕТ ЗАПУСКАТЬ ЧЕРЕЗ: python main.py
# ══════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    print()
    print("=" * 50)
    print("  🏆 Champions League AI Bot")
    print("  ✅ Сервер запускается...")
    print("  🌐 Открой браузер: http://localhost:8000")
    print("  🛑 Остановить: Ctrl + C")
    print("=" * 50)
    print()
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
