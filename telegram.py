import requests

def telegram_bot_sendtext(bot_message,bot_chatID):
    
    bot_token = '1066871713:AAGC9ZdGr675-2PWuJQcmtedZexGM0s_FXY'
    #bot_chatID = '369692491' , '193425555'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

    response = requests.get(send_text)

    return response.json()



#telegram_bot_sendtext('hi',"369692491")


