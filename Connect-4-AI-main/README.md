# Connect 4

A very strong machine learning based approach to Connect4, incorporating no prior game knowledge. Inspired by deepmind's alphazero. The implementation combines multiple different approaches to game solving, including min-maxing, monte carlo tree search, and machine learning (using a convolutional neural network) to classify/evaluate the strength of positions. 


https://user-images.githubusercontent.com/110444153/199854549-c7e782f9-0825-431c-bc1b-f735d6fe4bc4.mp4


# How it works:

The AI is shown a database of connect4 positions, and whether they resulted in a win, draw or loss. The AI is told to make a prediction as to what they think the result of a position would be, and then is told what the actual result was. In this way, it learns how to sense if a position is good or bad. 

The database of connect4 positions is made using monte carlo tree search, which (as a huge oversimplification) runs heaps of simulations for each possible move and records whether it wins or loses more on average to determine what moves are good. In this way, the AI is trained on data from quality players, and thus plays with quality itself. 

Once the AI can sense if a position is good or bad, in a real game we show the AI all possible futures for the next 7 moves and ask it to pick the future which it likes the most. This is done via minmaxing. 

# How to play against it yourself

Install all prerequisites in the requirements.txt. Python 3.8 was used (I believe).

Open CNNPlayWithSearch.py, run the program. The AI as it is currently built can only play first, but I believe also making it play second could actually be a very easy change. Type in your move when prompted (0 for first row, 6 for last row). 

To make the AI think shorter/longer you can go into CNNPlayWithSearch.py > heuristics > treeGen and then change counterL < whatever amount of positions you want the AI to see. The AI will increase amount of positions seen in powers of 7, so be careful with selection of large numbers.

# How to have it play against pure monte carlo tree search with no AI.

Open connect4.py, in connect4 > Step > find actionR variable and change it to actionR = monte(self.getState(),4000). You can change 4000 to any number to increase or decrease the strength of the monte carlo tree search player.

# How to train your own network

You need to first generate games. MonteGameGen.py creates games in a multiprocessing way. Currently the game generation is commented out. Uncomment the game generation part, and comment everything after. Generating games should take a long time and should make many files. You can stop the program at any time when you think theres enough games. These games are generated using just monteSearch of depth 1000, which you can change by scrolling up in MonteGameGen.py and changing the number from 1000 to something smaller if you want it to run faster. As a rule of thumb, you want about 200mb of data generated total. Sort these files into draws, wins and losses, then merge the CSV files (place each category of files into its own folder and use command line: copy *.csv merge.csv) into files called 0.csv , 1.csv and neg1.csv (rename merge.csv in each folder). Go to the end of 0.csv, 1.csv and neg1.csv and delete the last row which should contain an empty array (I think this is a bug as a result of multiprocessing, or maybe from merging)

Uncomment the second half of MonteGameGen.py and comment the first part that you just ran. Run it. This should create a file called final.csv which contains a large dataset of positions and their results. Now open the file Connect4CnnModel.py. This file should take a while to run as it deals with opening large files. Once your model is trained, you're good to go with running CNNPlayWithSearch.py 

# Pure Monte Gameplay

newMonte.py is an opponent using just monte carlo tree search, which is also interesting and can be played against. You're welcome to experiment with it.

# How you can contribute:

The code is far from clean, so if you would like to clean it / refactor that would be appreciated. Some fixes involve making it easier to vary the strength of your opponent and the type of your opponent (Monte, AI, human). A clickable version of the pygame window would also be lovely. Making the AI play second shouldnt be hard to fix, so thats a project for an enthusiast. Im happy to answer any questions anyone has, so feel free to contact me through github. If you run the code and theres any bugs which you had to fix, i can try fix it as well if you let me know. 
