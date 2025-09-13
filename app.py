import os
import time
import json
import threading
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

class RealTimeBitcoinTrader:
    """
    Twelve Data API yordamida BTC/USD narxini real vaqtda tahlil qilib,
    natijalarni Telegram'ga yuboruvchi va Telegram orqali boshqariladigan bot.
    """
    def __init__(self, api_key: str, telegram_token: str, chat_id: str):
        if not all([api_key, telegram_token, chat_id]):
            raise ValueError("Не все ключи или ID были предоставлены. Проверьте переменные окружения.")

        self.api_key = api_key
        self.telegram_token = telegram_token
        self.chat_id = str(chat_id)

        self.historical_data = self._generate_initial_historical_data()
        self.fib_levels = {}

        self.take_profit_percent = 3.0
        self.stop_loss_percent = 1.5

        self.stop_event = threading.Event()
        self.analysis_thread = None

        print("🤖 Real-Time Bitcoin Trading Bot sozlandi.")

    # --- Telegram отправка (через прямой HTTP, удобно из фонового потока) ---
    def _send_telegram_message(self, message_text: str):
        if not self.telegram_token or not self.chat_id:
            return
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {'chat_id': self.chat_id, 'text': message_text, 'parse_mode': 'Markdown'}
        try:
            requests.post(url, json=payload, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"❌ Telegram API bilan bog'lanishda xatolik: {e}")

    # --- Данные и индикаторы ---
    def _get_current_price(self) -> dict | None:
        """Twelve Data orqali joriy BTC/USD narxini oladi."""
        base_url = "https://api.twelvedata.com/price"
        params = {'symbol': 'BTC/USD', 'apikey': self.api_key, 'dp': 2}
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if 'price' in data and data['price'] is not None:
                return {'price': float(data['price']), 'timestamp': datetime.now()}
            else:
                return None
        except Exception as e:
            print(f"Narx olishda xatolik: {e}")
            return None

    def _generate_initial_historical_data(self, periods=200, interval_minutes=5) -> pd.DataFrame:
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=periods * interval_minutes)
        dates = pd.date_range(start=start_time, end=end_time, periods=periods)
        price = 65000 + np.random.uniform(-1000, 1000)
        prices = [price + 0.1 * np.sin(i / 50) + np.random.normal(0, 400) for i in range(periods)]
        df = pd.DataFrame(prices, index=dates, columns=['Close'])
        df['Open'] = df['Close'] - np.random.normal(0, 150, periods)
        df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 200, periods)
        df['Low']  = df[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 200, periods)
        return df

    def _update_historical_data(self, price_data: dict):
        now = price_data['timestamp']
        new_price = price_data['price']
        last_close = self.historical_data['Close'].iloc[-1]
        new_row = pd.DataFrame(
            {'Open': [last_close],
             'High': [max(last_close, new_price)],
             'Low':  [min(last_close, new_price)],
             'Close':[new_price]}, index=[now]
        )
        self.historical_data = pd.concat([self.historical_data, new_row])
        if len(self.historical_data) > 500:
            self.historical_data = self.historical_data.iloc[1:]

    def _calculate_fibonacci_levels(self, period=100):
        df = self.historical_data.tail(period)
        if len(df) < 20:
            self.fib_levels = {}
            return
        swing_high, swing_low = df['High'].max(), df['Low'].min()
        price_range = swing_high - swing_low
        if price_range == 0:
            self.fib_levels = {}
            return
        self.fib_levels = {
            'swing_high': swing_high, 'swing_low': swing_low,
            'level_23.6%': swing_high - price_range * 0.236,
            'level_38.2%': swing_high - price_range * 0.382,
            'level_50.0%': swing_high - price_range * 0.5,
            'level_61.8%': swing_high - price_range * 0.618,
        }

    def _calculate_indicators(self):
        df = self.historical_data
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        delta = df['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        self._calculate_fibonacci_levels()
        self.historical_data = df

    def _generate_signals(self) -> tuple[str, str]:
        df = self.historical_data.dropna()
        if len(df) < 2:
            return "KUTILMOQDA", "Yetarli ma'lumot yo'q"
        last, prev = df.iloc[-1], df.iloc[-2]

        # BUY signals
        if last['SMA_20'] > last['SMA_50'] and prev['SMA_20'] <= prev['SMA_50']:
            return "🟢 SOTIB OLISH (BUY)", "SMA 'Oltin kesishma' signali"
        if last['MACD'] > last['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
            return "🟢 SOTIB OLISH (BUY)", "MACD yuqoriga kesishmasi"
        if last['RSI'] > 30 and prev['RSI'] <= 30:
            return "🟢 SOTIB OLISH (BUY)", "RSI 'haddan tashqari sotilgan' zonadan chiqdi"
        if self.fib_levels and last['Close'] > last['SMA_50']:
            for level in [self.fib_levels['level_38.2%'],
                          self.fib_levels['level_50.0%'],
                          self.fib_levels['level_61.8%']]:
                if prev['Close'] < level and last['Close'] > level:
                    return "🟢 SOTIB OLISH (BUY)", f"Fibonachchi {level:.2f} darajasidan sakrash"

        # SELL signals
        if last['SMA_20'] < last['SMA_50'] and prev['SMA_20'] >= prev['SMA_50']:
            return "🔴 SOTISH (SELL)", "SMA 'O'lim kesishmasi' signali"
        if last['MACD'] < last['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
            return "🔴 SOTISH (SELL)", "MACD pastga kesishmasi"
        if last['RSI'] < 70 and prev['RSI'] >= 70:
            return "🔴 SOTISH (SELL)", "RSI 'haddan tashqari olingan' zonadan chiqdi"

        return "⚪ KUTISH (HOLD)", "Kuchli signal mavjud emas"

    def analysis_loop(self, interval_seconds=120):
        """Alohida thread ichida tahlil."""
        cycle_count = 0
        while not self.stop_event.is_set():
            try:
                cycle_count += 1
                price_data = self._get_current_price()
                if not price_data:
                    self.stop_event.wait(interval_seconds)
                    continue

                self._update_historical_data(price_data)
                self._calculate_indicators()
                signal, reason = self._generate_signals()

                message_lines = [
                    f"🔄 *Tahlil Sikli #{cycle_count} | {datetime.now().strftime('%H:%M:%S')}*",
                    f"💰 Joriy BTC/USD narxi: `${price_data['price']:.2f}`",
                    "\n📊 *BASHORAT:*",
                    f"   Signal: {signal}",
                    f"   Sabab: {reason}",
                ]

                if "SOTIB OLISH" in signal or "SOTISH" in signal:
                    entry_price = price_data['price']
                    if "SOTIB OLISH" in signal:
                        tp = entry_price * (1 + self.take_profit_percent / 100)
                        sl = entry_price * (1 - self.stop_loss_percent / 100)
                        message_lines.append("\n🎯 *SAVDO DARAJALARI (BUY):*")
                    else:
                        tp = entry_price * (1 - self.take_profit_percent / 100)
                        sl = entry_price * (1 + self.stop_loss_percent / 100)
                        message_lines.append("\n🎯 *SAVDO DARAJALARI (SELL):*")
                    message_lines += [
                        f"   Take Profit: `${tp:.2f}`",
                        f"   Stop Loss: `${sl:.2f}`",
                    ]

                telegram_message = "\n".join(message_lines)
                print("\n" + telegram_message.replace('*', '').replace('`', ''))
                self._send_telegram_message(telegram_message)
            except Exception as e:
                print(f"Loop error: {e}")

            self.stop_event.wait(interval_seconds)

    # --- TELEGRAM КОМАНДЫ (async) ---
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_chat_id = str(update.effective_chat.id)
        if user_chat_id != self.chat_id:
            await context.bot.send_message(chat_id=user_chat_id, text="Sizda bu botni boshqarish huquqi yo'q.")
            return

        if self.analysis_thread and self.analysis_thread.is_alive():
            await context.bot.send_message(chat_id=self.chat_id, text="⚠️ Bot allaqachon ishlamoqda!")
        else:
            await context.bot.send_message(chat_id=self.chat_id, text="✅ Tahlil sikli boshlandi. Natijalar har 2 daqiqada yuboriladi.")
            self.stop_event.clear()
            self.analysis_thread = threading.Thread(target=self.analysis_loop, daemon=True)
            self.analysis_thread.start()

    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_chat_id = str(update.effective_chat.id)
        if user_chat_id != self.chat_id:
            await context.bot.send_message(chat_id=user_chat_id, text="Sizda bu botni boshqarish huquqi yo'q.")
            return

        if self.analysis_thread and self.analysis_thread.is_alive():
            await context.bot.send_message(chat_id=self.chat_id, text="🛑 Tahlil sikli to'xtatilmoqda...")
            self.stop_event.set()
            # Не блокируем event loop долгим join; пусть поток сам завершится
            await context.bot.send_message(chat_id=self.chat_id, text="✅ Stop signali berildi. Bir necha soniya ichida to'xtaydi.")
        else:
            await context.bot.send_message(chat_id=self.chat_id, text="⚠️ Bot hozirda ishlamayapti.")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.analysis_thread and self.analysis_thread.is_alive():
            await context.bot.send_message(chat_id=update.effective_chat.id, text="ℹ️ Holat: Bot faol va tahlil qilmoqda.")
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="ℹ️ Holat: Bot passiv. /start buyrug'i bilan ishga tushiring.")

if __name__ == "__main__":
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")  # str(chat_id)
    # Поддержим оба названия ключа, но лучше использовать TWELVEDATA_API_KEY
    TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY") or os.getenv("YOUR_API_KEY")

    # Для webhook нужен внешний адрес твоей веб-службы на Render
    APP_BASE_URL = os.getenv("APP_BASE_URL")  # например: https://telegram-crypto-bot.onrender.com
    WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "telegram")  # путь эндпойнта (можно поменять)
    SECRET_TOKEN = os.getenv("TELEGRAM_WEBHOOK_SECRET")   # опционально: секрет вебхука

    # Порт Render передаёт в переменной PORT
    PORT = int(os.getenv("PORT", "8000"))

    missing = []
    for k, v in {
        "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
        "TWELVEDATA_API_KEY": TWELVEDATA_API_KEY,
        "APP_BASE_URL": APP_BASE_URL,
    }.items():
        if not v:
            missing.append(k)
    if missing:
        print("❌ Отсутствуют переменные окружения:", ", ".join(missing))
        raise SystemExit(1)

    bot_instance = RealTimeBitcoinTrader(
        api_key=TWELVEDATA_API_KEY,
        telegram_token=TELEGRAM_BOT_TOKEN,
        chat_id=TELEGRAM_CHAT_ID
    )

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", bot_instance.start_command))
    application.add_handler(CommandHandler("stop", bot_instance.stop_command))
    application.add_handler(CommandHandler("status", bot_instance.status_command))

    public_webhook_url = f"{APP_BASE_URL.rstrip('/')}/{WEBHOOK_PATH}"
    print("🚀 Telegram bot buyruqlarni eshitishni boshladi (webhook)...")
    print(f"🌐 Webhook URL: {public_webhook_url}")
    print(f"🔉 Listening on 0.0.0.0:{PORT} path=/{WEBHOOK_PATH}")

    # Запуск вебхука (PTB сам выставит setWebhook на указанный URL)
    application.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=WEBHOOK_PATH,
        webhook_url=public_webhook_url,
        secret_token=SECRET_TOKEN,          # можешь удалить, если не используешь
        allowed_updates=Update.ALL_TYPES,
    )

