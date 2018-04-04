import requests
from bs4 import BeautifulSoup

def code(heart):
    url = 'https://www.etymonline.com/word/' + str(heart)  # sets url page number
    source_code = requests.get(url)  # captures all the html of url
    plain_text = source_code.text  # converts the html into a text file
    soup = BeautifulSoup(plain_text, 'lxml')  # enters plaint_text into BeautifulSoup class as the object 'soup'
    # makes it possible to apply the functions of beautiful soup on plain_text
    #for string in soup.findALL('li', {'class': 'related__word--3Si0N'}):  # searches for the 'class' called 'item-name'
        #secret = string.get('href')  # sets href = to a url for the objects we searched for
        # title = link.string    #sets title = the plaintext of the object we searched for
    secret = str(soup.find("li", class_="related__word--3Si0N"))
    secret = secret[33:]
    secret = secret[:-5]
    return secret

while True:
    search = raw_input("reveal or quit: ")
    if search == "quit":
        print "closing...."
        break

    key = str(code(search))
    if key == "":
        print ("_absence_discovered_")
    else:
        print (key)
