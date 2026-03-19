package bk;

import game.Game;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import main.collections.FastArrayList;
import other.AI;
import other.context.Context;
import other.move.Move;
import other.state.container.ContainerState;

/**
 * Wrapper Ludii pour le joueur Alpha-Beta Hex 11x11 (TC_MG_alphabeta_hex).
 *
 * Convention Ludii pour Hex 11x11 :
 *   - Cases numérotées 0..120, ligne par ligne de haut en bas.
 *   - Joueur 1 (playerID=1) = Blue (O) : connecte Nord→Sud
 *   - Joueur 2 (playerID=2) = Red  (@) : connecte Ouest→Est
 */
public class TC_MG_alphabeta_hex extends AI {

    public static final int EMPTY = 0;
    public static final int BLUE  = 1;
    public static final int RED   = 2;

    public static final int HEX_SIZE  = 11;
    public static final int HEX_CELLS = 121;

    public static final String local_player_str = "../TC_MG_alphabeta_hex";

    protected int player = -1;

    public TC_MG_alphabeta_hex() {
        this.friendlyName = "TC_MG_alphabeta_hex";
    }

    @Override
    public Move selectAction(
        final Game game,
        final Context context,
        final double maxSeconds,
        final int maxIterations,
        final int maxDepth
    ) {
        FastArrayList<Move> legalMoves = game.moves(context).moves();

        int[] board = new int[HEX_CELLS];
        for (final ContainerState cs : context.state().containerStates()) {
            for (int i = 0; i < HEX_CELLS; i++) {
                if (cs.isEmptyCell(i))        board[i] = EMPTY;
                else if (cs.whoCell(i) == 1)  board[i] = BLUE;
                else if (cs.whoCell(i) == 2)  board[i] = RED;
            }
        }

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < HEX_CELLS; i++) {
            if      (board[i] == BLUE) sb.append('O');
            else if (board[i] == RED)  sb.append('@');
            else                       sb.append('.');
        }

        String turn = (player == 1) ? "O" : "@";

        String res = "";
        try {
            Process process = startProcess(local_player_str, sb.toString(), turn);
            res = readProcess(process);
        } catch (Exception e) {
            e.printStackTrace();
        }

        if (res.length() >= 2) {
            int col = res.charAt(0) - 'A';
            int row = -1;
            try {
                row = Integer.parseInt(res.substring(1)) - 1;
            } catch (NumberFormatException e) {
                System.err.println("TC_MG_alphabeta_hex: cannot parse move: " + res);
            }
            if (col >= 0 && col < HEX_SIZE && row >= 0 && row < HEX_SIZE) {
                int pos = row * HEX_SIZE + col;
                for (int i = 0; i < legalMoves.size(); i++) {
                    if (legalMoves.get(i).to() == pos) {
                        return legalMoves.get(i);
                    }
                }
                System.err.println("TC_MG_alphabeta_hex: move " + res + " (pos=" + pos + ") not in legal moves");
            }
        } else {
            System.err.println("TC_MG_alphabeta_hex: unexpected response: '" + res + "'");
        }

        System.err.println("TC_MG_alphabeta_hex: fallback to first legal move");
        return legalMoves.get(0);
    }

    @Override
    public void initAI(final Game game, final int playerID) {
        this.player = playerID;
    }

    private Process startProcess(String exe, String board, String player) throws IOException {
        ProcessBuilder pb = new ProcessBuilder(exe, board, player);
        pb.redirectErrorStream(false);
        try {
            return pb.start();
        } catch (Exception e) {
            System.err.println("TC_MG_alphabeta_hex: cannot start process: " + e);
        }
        return null;
    }

    private String readProcess(Process process) throws IOException, InterruptedException {
        if (process == null) return "";
        StringBuilder out = new StringBuilder();
        StringBuilder err = new StringBuilder();
        try (BufferedReader r = new BufferedReader(
                new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = r.readLine()) != null) out.append(line);
            process.waitFor();
        } catch (Exception e) { System.err.println(e); }
        try (BufferedReader r = new BufferedReader(
                new InputStreamReader(process.getErrorStream()))) {
            String line;
            while ((line = r.readLine()) != null) err.append(line).append('\n');
            process.waitFor();
        } catch (Exception e) { System.err.println(e); }
        String errStr = err.toString().trim();
        if (!errStr.isEmpty()) System.err.println("[alphabeta_hex] " + errStr);
        return out.toString().trim();
    }
}
