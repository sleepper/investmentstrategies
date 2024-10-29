import logging
from openai import OpenAI


from telegram import Update, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, filters, MessageHandler, InlineQueryHandler

# Set up the logger

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Set up your API key
client = OpenAI()

def business_description(str_company_name:str):

    # Define the model and send the request
    response = client.chat.completions.create(
    model="gpt-4o-mini",  # Specify the model
    messages=[
            {"role": "system", "content": "You are an investment banking associate, whose job is to give short, insightful and correct answers."},
            {"role": "user", "content": f"Write me a business description for the company {str_company_name}. The answer should include the a high level business description, examples of products and their business applications, geographical presense of the company, major shareholders."}
        ],
    temperature = 0.5,
    max_tokens = 300
    )

    return response.choices[0].message.content

# Function to process the specific type of update, i.e. /start update
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")

# Function to process the specific type of update, i.e. conversations
async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)

# Function to process the specific type of update, i.e. capitilize the incoming text
async def caps(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_caps = ' '.join(context.args).upper()
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text_caps)

# My own conversational function. Take the text from the message and use with ChatGPT for the reply
async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    str_ticker = update.message.text
    str_reply = business_description(str_ticker)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=str_reply)

# Inline version of the function
async def inline_caps(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.inline_query.query
    if not query:
        return
    results = []
    results.append(
        InlineQueryResultArticle(
            id=query.upper(),
            title='caps',
            input_message_content=InputTextMessageContent(query.upper())
        )
    )
    await context.bot.answer_inline_query(update.inline_query.id, results)

# Function to process the unknown commands
async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.")

if __name__ == '__main__':
    
    # initiate the bot
    application = ApplicationBuilder().token('1470074102:AAEd4en4YhzADfW2mgMb7m4Xp0_HHKoD_qw').build()
    
    # create different types of handlers
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)
    
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)
    application.add_handler(echo_handler)

    caps_handler = CommandHandler('caps',caps)
    application.add_handler(caps_handler)

    inline_caps_handler = InlineQueryHandler(inline_caps)
    application.add_handler(inline_caps_handler)

    unknown_handler = MessageHandler(filters.COMMAND, unknown)
    application.add_handler(unknown_handler)

    application.run_polling()