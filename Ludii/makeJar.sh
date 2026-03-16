rm -f TC_MG_alphabeta_player.jar TC_MG_montecarlo.jar TC_MG_mcts.jar
javac -cp Ludii-1.3.14.jar TC_MG_alphabeta_player.java
mv TC_MG_alphabeta_player.class bk/
jar cf TC_MG_alphabeta_player.jar bk/TC_MG_alphabeta_player.class

javac -cp Ludii-1.3.14.jar TC_MG_montecarlo.java
mv TC_MG_montecarlo.class bk/
jar cf TC_MG_montecarlo.jar bk/TC_MG_montecarlo.class

javac -cp Ludii-1.3.14.jar TC_MG_mcts.java
mv TC_MG_mcts.class bk/
jar cf TC_MG_mcts.jar bk/TC_MG_mcts.class
