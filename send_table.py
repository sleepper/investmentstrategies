import tabulate
import asyncio
import telegram

import pandas as pd

from telegram import BotCommand

data = {
        'Name':['Alex','Alexander'],
        'Age':[1,31]
    }

df = pd.DataFrame(data).to_string(index=False)
df = '<pre>' + df + '</pre>'

async def main():
    bot = telegram.Bot('1470074102:AAEd4en4YhzADfW2mgMb7m4Xp0_HHKoD_qw')
    async with bot:
        await bot.set_my_commands([BotCommand('start','start_text'),BotCommand('help','help_text'),BotCommand('text','text_text'),BotCommand('picture','picture_text')])
        await bot.send_message(text=df, chat_id = '145125445', parse_mode='HTML')
        print(df)

if __name__ == '__main__':
    asyncio.run(main())