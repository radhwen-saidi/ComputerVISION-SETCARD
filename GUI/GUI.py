import tkinter as tk
from PIL import Image, ImageTk
import cv2


LOGO_PATH = "SET.png"
HOME_H,HOME_W = 853,480
HOW_H,HOW_W =480,480
PLAYER_S_H,PLAYER_S_W=600,480


class Game:

    player1=None
    player2=None
    

    def Launch(self):
        Home(self)
    

        
class Player:
    name = None
    score = 0

    def __init__(self,name,score):
        self.name = name
        self.score = score

    def SetScore(self,score):
        self.score=score
    
    def getScore(self):
        return self.score

    def SetName(self,name):
        self.name = name

    def getName(self):
        return self.name

class Home:

    Game = None
    window = None
    
    def __init__(self,Game):

        self.Game = Game

        window = tk.Tk()
        self.window = window
        canvas = tk.Canvas(window,width=HOME_W,height=HOME_H)
        canvas.grid(columnspan=3,rowspan=4)

        logo = Image.open(LOGO_PATH)
        logo = logo.resize((300,150))
        logo = ImageTk.PhotoImage(logo)
        
        logo_label = tk.Label(image=logo)
        logo_label.image = logo
        logo_label.grid(column=1,row=0)

        instructions = tk.Label(window,text="The Family Game Of Visual Perception",font="Raleway")
        instructions.grid(columnspan=3,column=0,row=1)

        play_text = tk.StringVar()
        play_text.set("Play")
        play_button = tk.Button(window,textvariable =play_text,command = lambda:self.playerSelection(),font ="Raleway",bg="#7385AE",height=2,width=20)
        play_button.grid(column=1,row=2)

        How_to_text = tk.StringVar()
        How_to_text.set("How To Play")
        How_to_button = tk.Button(window,textvariable =How_to_text,command = lambda:self.howToPlay(), font ="Raleway",bg="#7385AE",height=2,width=20)
        How_to_button.grid(column=1,row=3)


        window.mainloop()

    def playerSelection(self):
        self.window.destroy()
        select = PlayerSelection(self.Game)
        self.window = select.window

    def howToPlay(self):
        self.window.destroy()
        how = HowToPlay()
        self.window=how.window


#Add a button to go back home in instructions !!

class HowToPlay:

    
    window = None

    def __init__(self):
        window = tk.Tk()
        self.window=window
        canvas = tk.Canvas(window,width=HOW_W,height=HOW_H)
        canvas.grid(columnspan=3,rowspan=4)

        logo = Image.open(LOGO_PATH)
        logo = logo.resize((200,100))
        logo = ImageTk.PhotoImage(logo)

        logo_label = tk.Label(image=logo)
        logo_label.image = logo
        logo_label.grid(column=1,row=0)

        instructions = tk.Label(window,text="Instructions en cours d'ajout ...",font="Raleway")
        instructions.grid(columnspan=3,column=0,row=1)


class PlayerSelection:
    
    window = None
    name1= None
    name2= None
    Game = None
    
    def __init__(self,Game):
        
        self.Game = Game
        window = tk.Tk()
        self.window=window
        canvas = tk.Canvas(window,width=PLAYER_S_W,height=PLAYER_S_H)
        canvas.grid(columnspan=3,rowspan=6)

        logo = Image.open(LOGO_PATH)
        logo = logo.resize((200,100))
        logo = ImageTk.PhotoImage(logo)

        logo_label = tk.Label(image=logo)
        logo_label.image = logo
        logo_label.grid(column=1,row=0)


        lab1 = tk.Label(window,text="Player 1 :",font="Raleway")
        lab1.grid(columnspan=3,column=0,row=1)

        self.name1 = tk.StringVar()
        self.name1.set("Player Name")
        entree = tk.Entry(window, textvariable=self.name1, width=30)
        entree.grid(columnspan=3,column=0,row=2)

        lab2 = tk.Label(window,text="Player 2 :",font="Raleway")
        lab2.grid(columnspan=3,column=0,row=3)

        self.name2 = tk.StringVar()
        self.name2.set("Player Name")
        entree2 = tk.Entry(window, textvariable=self.name2, width=30)
        entree2.grid(columnspan=3,column=0,row=4)


        How_to_text = tk.StringVar()
        How_to_text.set("Go!")
        How_to_button = tk.Button(window,textvariable =How_to_text,command = lambda:self.startGame(self.Game), font ="Raleway",bg="#7385AE",height=2,width=20)
        How_to_button.grid(column=1,row=5)
        
    def startGame(self,Game):
        #print("player 1 : {}".format(self.name1.get()))
        #print("player 2 : {}".format(self.name2.get()))
        player1 = Player(self.name1.get(),0)
        player2 = Player(self.name2.get(),0)

        print("player 1: {}".format(player1.getName()))
        print("player 2: {}".format(player2.getName()))

#class GameWindow:
class GameWindow:

    window = None

    def __init__(self):
        self.window = tk.Tk()



g = Game()
g.Launch()