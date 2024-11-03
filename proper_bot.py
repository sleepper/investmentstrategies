import pandas as pd
import logging
from telegram import Update, BotCommand
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, Updater

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

#https://servat.medium.com/creating-a-telegram-bot-with-python-4a3b4906c101

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Hello, this bot is your personal financial advisor with all the relevant disclaimers")

async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="This is how the bot works") #TODO Add HTML text here

async def send_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_says = context.args
    await context.bot.send_message(chat_id=update.effective_chat.id, text = f"Below you will find the business description for {user_says}. \nbla-bla-bla")

async def send_table(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = {
            'Name':['Alex','Alexander'],
            'Age':[1,31]
        }
    df_str = pd.DataFrame(data).to_string(index=False)
    df_str_tm = '<pre>' + df_str + '</pre>'
    await context.bot.send_message(chat_id=update.effective_chat.id, text = df_str_tm, parse_mode='HTML')

async def send_picture(update: Update, context: ContextTypes.DEFAULT_TYPE):
    str_path = 'profiles\ADBE\_stats_profile.png'
    await context.bot.send_document(chat_id=update.effective_chat.id, document = open(str_path,'rb'))

if __name__ == '__main__':
    
    application = ApplicationBuilder().token('1470074102:AAEd4en4YhzADfW2mgMb7m4Xp0_HHKoD_qw').build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help))
    application.add_handler(CommandHandler('text', send_text))
    application.add_handler(CommandHandler('table', send_table))
    application.add_handler(CommandHandler('picture', send_picture))

    #application.bot.set_my_commands([BotCommand('start','start_text'),BotCommand('help','help_text'),BotCommand('text','text_text')])

    application.run_polling()