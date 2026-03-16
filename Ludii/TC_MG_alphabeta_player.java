package bk;

import java.util.concurrent.ThreadLocalRandom;

import game.Game;
import main.collections.FastArrayList;
import other.AI;
import other.context.Context;
import other.state.container.ContainerState;
import other.move.Move;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Map;
import java.util.HashMap;

public class TC_MG_alphabeta_player extends AI
{		
	public final static int EMPTY = 0;
	public final static int BLACK = 1;
	public final static int WHITE = 2;
	//public final static String local_player_str = "/Users/n/LudiiPlayers/rand_player";
	public final static String local_player_str = "../TC_MG_alphabeta_player";

	protected int player = -1; // player_index
	
	public TC_MG_alphabeta_player()	{
			this.friendlyName = "TC_MG_alphabeta_player";
	}
	
	@Override
	public Move selectAction
	(
		final Game game, 
		final Context context, 
		final double maxSeconds,
		final int maxIterations,
		final int maxDepth
	)
	{
		FastArrayList<Move> legalMoves = game.moves(context).moves();
    int board[] = new int[64];
		for (final ContainerState containerState : context.state().containerStates()) {
			for(int i = 0; i < 64; i++) {
				if(containerState.isEmptyCell(i)) board[i] = EMPTY;
				if(containerState.whoCell(i) == 2) board[i] = BLACK;
				if(containerState.whoCell(i) == 1) board[i] = WHITE;
			}
		}
		StringBuilder sb = new StringBuilder();
		for(int i = 7; i >= 0; i--) {
			for(int j = 0; j < 8; j++) {
				if(board[i*8+j] == BLACK) {sb.append("@"); }
				else if(board[i*8+j] == WHITE) {sb.append("O"); }
				else { sb.append("."); }
			}
		}
		String turn = "O";
		if(player==2) turn = "@";
		String res = "";		
		int line_i = -1;
		int col_i = -1;
		int line_f = -1;
		int col_f = -1;
		try {
			  //System.out.println("[info] "+sb.toString()+" "+turn);
				Process process = startProcessLocal(local_player_str,sb.toString(), turn);
				res = endProcessLocal(process);
				if(res.length()==5) {
					//System.out.println("[info] "+local_player_str+" play "+res);
	  	  	col_i = res.charAt(0)-'A';
  		  	line_i = res.charAt(1)-'1';
		    	col_f = res.charAt(3)-'A';
		    	line_f = res.charAt(4)-'1';
    			//System.err.println("= ("+line_i+" "+col_i+") ("+line_f+" "+col_f+")");
				} else {
					System.err.println("error res.length() "+res.length());
				}
    } 
	catch(Exception e) { 
      e.printStackTrace();
    }
		if(line_i!=-1 && col_i !=-1 && line_f!=-1 && col_f!=-1) {
  		int pos_i = line_i*8+col_i;
	  	int pos_f = line_f*8+col_f;
			char ccoli = (char)('A'+col_i);
			char ccolf = (char)('A'+col_f);
    	//System.err.println(local_player_str+" play ["+pos_i+" "+pos_f+"] "+(line_i+1)+ccoli+"-"+(line_f+1)+ccolf);
		  for(int i = 0; i < legalMoves.size(); i++) {
			  if(legalMoves.get(i).from() == pos_i && legalMoves.get(i).to() == pos_f) {return legalMoves.get(i);}
		  }
		}	  
		System.out.println(local_player_str+" play random");
		return legalMoves.get(0);
	}
	
	@Override
	public void initAI(final Game game, final int playerID)
	{
		this.player = playerID;
	}	
	public Process startProcessLocal(String _player_str, String _b, String _turn) throws IOException
	{
		ProcessBuilder processBuilder = new ProcessBuilder(_player_str,_b,_turn);
		processBuilder.redirectErrorStream(false);
		try {
			return processBuilder.start();
		} catch(Exception e) { System.err.println(e); }
		return null;
	}
	public String endProcessLocal(Process process) throws IOException, InterruptedException 
  {
		StringBuilder processOutput = new StringBuilder();
		StringBuilder processErr = new StringBuilder();
		try (BufferedReader processOutputReader = new BufferedReader(
						new InputStreamReader(process.getInputStream()));) {
				String readLine;
				while ((readLine = processOutputReader.readLine()) != null) {
						processOutput.append(readLine + System.lineSeparator());
				}
				process.waitFor();
		} catch(Exception e) { System.err.println(e); } 
		try (BufferedReader processOutputReader = new BufferedReader(
						new InputStreamReader(process.getErrorStream()));) {
				String readLine;
				while ((readLine = processOutputReader.readLine()) != null) {
						processErr.append(readLine + System.lineSeparator());
				}
				process.waitFor();
		} catch(Exception e) { System.err.println(e); } 		
		//System.err.println(processErr.toString().trim());
		//System.err.println("getProcessOutput end ---\n");
		return processOutput.toString().trim();
	}
}
