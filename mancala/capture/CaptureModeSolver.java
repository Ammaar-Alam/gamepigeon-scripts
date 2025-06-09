import java.util.Scanner;
import java.util.ArrayList;
import java.util.List;

public class CaptureModeSolver {
    private static final int BOARD_SIZE = 14;
    private static final int PLAYER1_STORE = 6;
    private static final int PLAYER2_STORE = 13;
    private static final int MAX_DEPTH = 12;
    
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        
        System.out.println("=== Mancala CAPTURE MODE Solver ===");
        System.out.println("This solver calculates the BEST moves to win every time!");
        System.out.println();
        
        System.out.print("Is it YOUR turn to play? (y/n): ");
        boolean isPlayerTurn = input.next().toLowerCase().charAt(0) == 'y';
        input.nextLine();
        
        System.out.println("\nEnter the current board state:");
        System.out.println("Input stones in each pit starting from YOUR top-left, going counter-clockwise.");
        System.out.println("Enter 14 numbers total: your 6 pits, your store, opponent's 6 pits, opponent's store");
        System.out.println("Example: 4 4 4 4 4 4 0 4 4 4 4 4 4 0");
        
        int[] board = new int[BOARD_SIZE];
        for (int i = 0; i < BOARD_SIZE; i++) {
            board[i] = input.nextInt();
        }
        
        System.out.println("\nCalculating best moves...\n");
        
        GameState state = new GameState(board, isPlayerTurn);
        analyzeBestMoves(state);
        
        input.close();
    }
    
    private static void analyzeBestMoves(GameState state) {
        System.out.println("Current Board:");
        printBoard(state.board);
        System.out.println();
        
        List<Move> bestMoves = new ArrayList<>();
        int moveCount = 0;
        GameState currentState = state;
        
        while (!currentState.isGameOver() && moveCount < 20) {
            Move bestMove = findBestMove(currentState);
            if (bestMove == null) break;
            
            bestMoves.add(bestMove);
            moveCount++;
            
            System.out.println("\n=== Move " + moveCount + " ===");
            System.out.println((currentState.isPlayerTurn ? "Your" : "Opponent's") + 
                              " best move: Pit " + (bestMove.pit + 1) + 
                              " (Score advantage: " + bestMove.score + ")");
            
            // Check if this move would empty the player's side
            GameState nextState = makeMove(currentState, bestMove.pit);
            if (currentState.isPlayerTurn) {
                boolean playerSideEmpty = true;
                for (int i = 0; i < 6; i++) {
                    if (nextState.board[i] > 0) {
                        playerSideEmpty = false;
                        break;
                    }
                }
                if (playerSideEmpty && !nextState.isGameOver()) {
                    System.out.println("WARNING: This move empties your side!");
                }
            }
            
            currentState = nextState;
            
            System.out.println("\nBoard after move:");
            printBoard(currentState.board);
            System.out.println("Scores - You: " + currentState.board[PLAYER1_STORE] + 
                              ", Opponent: " + currentState.board[PLAYER2_STORE]);
            
            if (moveCount >= 10 && currentState.isPlayerTurn) {
                System.out.println("\nShowing first 10 of your moves. Continue playing optimally!");
                break;
            }
        }
        
        if (currentState.isGameOver()) {
            System.out.println("\n=== GAME OVER ===");
            
            // Check why game ended
            boolean playerSideEmpty = isGameSideEmpty(currentState.board, 0, 6);
            boolean opponentSideEmpty = isGameSideEmpty(currentState.board, 7, 13);
            
            if (playerSideEmpty) {
                System.out.println("Your side is empty - opponent collects remaining stones!");
            } else if (opponentSideEmpty) {
                System.out.println("Opponent's side is empty - you collect remaining stones!");
            }
            
            System.out.println("Final Score - You: " + currentState.board[PLAYER1_STORE] + 
                              ", Opponent: " + currentState.board[PLAYER2_STORE]);
            if (currentState.board[PLAYER1_STORE] > currentState.board[PLAYER2_STORE]) {
                System.out.println("YOU WIN!");
            } else if (currentState.board[PLAYER1_STORE] < currentState.board[PLAYER2_STORE]) {
                System.out.println("OPPONENT WINS!");
            } else {
                System.out.println("IT'S A TIE!");
            }
        }
    }
    
    private static Move findBestMove(GameState state) {
        int bestScore = state.isPlayerTurn ? Integer.MIN_VALUE : Integer.MAX_VALUE;
        int bestPit = -1;
        boolean extraTurn = false;
        
        List<Integer> validMoves = getValidMoves(state);
        if (validMoves.isEmpty()) return null;
        
        for (int pit : validMoves) {
            GameState newState = makeMove(state, pit);
            
            boolean getsExtraTurn = (newState.isPlayerTurn == state.isPlayerTurn);
            
            int score = minimax(newState, MAX_DEPTH - 1, Integer.MIN_VALUE, Integer.MAX_VALUE, state.isPlayerTurn);
            
            if (state.isPlayerTurn) {
                if (score > bestScore) {
                    bestScore = score;
                    bestPit = pit;
                    extraTurn = getsExtraTurn;
                }
            } else {
                if (score < bestScore) {
                    bestScore = score;
                    bestPit = pit;
                    extraTurn = getsExtraTurn;
                }
            }
        }
        
        if (bestPit == -1) return null;
        return new Move(bestPit, bestScore, extraTurn);
    }
    
    private static int minimax(GameState state, int depth, int alpha, int beta, boolean originalPlayer) {
        if (depth == 0 || state.isGameOver()) {
            return evaluateBoard(state, originalPlayer);
        }
        
        List<Integer> validMoves = getValidMoves(state);
        if (validMoves.isEmpty()) {
            return evaluateBoard(state, originalPlayer);
        }
        
        if (state.isPlayerTurn == originalPlayer) {
            int maxEval = Integer.MIN_VALUE;
            for (int pit : validMoves) {
                GameState newState = makeMove(state, pit);
                int eval = minimax(newState, depth - 1, alpha, beta, originalPlayer);
                maxEval = Math.max(maxEval, eval);
                alpha = Math.max(alpha, eval);
                if (beta <= alpha) break;
            }
            return maxEval;
        } else {
            int minEval = Integer.MAX_VALUE;
            for (int pit : validMoves) {
                GameState newState = makeMove(state, pit);
                int eval = minimax(newState, depth - 1, alpha, beta, originalPlayer);
                minEval = Math.min(minEval, eval);
                beta = Math.min(beta, eval);
                if (beta <= alpha) break;
            }
            return minEval;
        }
    }
    
    private static GameState makeMove(GameState state, int pit) {
        int[] newBoard = state.board.clone();
        boolean isPlayerTurn = state.isPlayerTurn;
        
        int stones = newBoard[pit];
        newBoard[pit] = 0;
        
        int currentPit = pit;
        while (stones > 0) {
            currentPit = (currentPit + 1) % BOARD_SIZE;
            
            if ((isPlayerTurn && currentPit == PLAYER2_STORE) || 
                (!isPlayerTurn && currentPit == PLAYER1_STORE)) {
                continue;
            }
            
            newBoard[currentPit]++;
            stones--;
        }
        
        boolean extraTurn = false;
        if ((isPlayerTurn && currentPit == PLAYER1_STORE) || 
            (!isPlayerTurn && currentPit == PLAYER2_STORE)) {
            extraTurn = true;
        }
        
        if (!extraTurn && currentPit != PLAYER1_STORE && currentPit != PLAYER2_STORE) {
            if (newBoard[currentPit] == 1) {
                int oppositePit = 12 - currentPit;
                if ((isPlayerTurn && currentPit >= 0 && currentPit < 6) ||
                    (!isPlayerTurn && currentPit >= 7 && currentPit < 13)) {
                    if (newBoard[oppositePit] > 0) {
                        int capturedStones = newBoard[oppositePit] + 1;
                        newBoard[oppositePit] = 0;
                        newBoard[currentPit] = 0;
                        
                        if (isPlayerTurn) {
                            newBoard[PLAYER1_STORE] += capturedStones;
                        } else {
                            newBoard[PLAYER2_STORE] += capturedStones;
                        }
                    }
                }
            }
        }
        
        if (isGameSideEmpty(newBoard, 0, 6) || isGameSideEmpty(newBoard, 7, 13)) {
            collectRemainingStones(newBoard);
        }
        
        return new GameState(newBoard, extraTurn ? isPlayerTurn : !isPlayerTurn);
    }
    
    private static List<Integer> getValidMoves(GameState state) {
        List<Integer> moves = new ArrayList<>();
        int start = state.isPlayerTurn ? 0 : 7;
        int end = state.isPlayerTurn ? 6 : 13;
        
        for (int i = start; i < end; i++) {
            if (state.board[i] > 0) {
                moves.add(i);
            }
        }
        return moves;
    }
    
    private static int evaluateBoard(GameState state, boolean forPlayer) {
        int playerScore = state.board[PLAYER1_STORE];
        int opponentScore = state.board[PLAYER2_STORE];
        
        int playerStones = 0;
        for (int i = 0; i < 6; i++) {
            playerStones += state.board[i];
        }
        
        int opponentStones = 0;
        for (int i = 7; i < 13; i++) {
            opponentStones += state.board[i];
        }
        
        // Check if game is over - if so, calculate final scores
        if (playerStones == 0 || opponentStones == 0) {
            playerScore += playerStones;
            opponentScore += opponentStones;
            
            // Game over - return win/loss/tie evaluation
            int finalDiff = playerScore - opponentScore;
            if (finalDiff > 0) {
                return forPlayer ? 100000 : -100000;  // Player wins
            } else if (finalDiff < 0) {
                return forPlayer ? -100000 : 100000;  // Opponent wins
            } else {
                return 0;  // Tie
            }
        }
        
        int scoreDiff = playerScore - opponentScore;
        int stonesDiff = playerStones - opponentStones;
        
        int evaluation = scoreDiff * 100 + stonesDiff * 10;
        
        // Bonus for potential captures
        for (int i = 0; i < 6; i++) {
            if (state.board[i] > 0 && state.board[12 - i] > 0) {
                evaluation += state.board[12 - i] * 2;
            }
        }
        
        // Penalty for having too few stones (risk of emptying side)
        if (playerStones <= 2) {
            evaluation -= (3 - playerStones) * 50;
        }
        
        return forPlayer ? evaluation : -evaluation;
    }
    
    private static boolean isGameSideEmpty(int[] board, int start, int end) {
        for (int i = start; i < end; i++) {
            if (board[i] > 0) return false;
        }
        return true;
    }
    
    private static void collectRemainingStones(int[] board) {
        int playerStones = 0;
        for (int i = 0; i < 6; i++) {
            playerStones += board[i];
            board[i] = 0;
        }
        board[PLAYER1_STORE] += playerStones;
        
        int opponentStones = 0;
        for (int i = 7; i < 13; i++) {
            opponentStones += board[i];
            board[i] = 0;
        }
        board[PLAYER2_STORE] += opponentStones;
    }
    
    private static void printBoard(int[] board) {
        System.out.println("  " + board[12] + " " + board[11] + " " + board[10] + " " + 
                          board[9] + " " + board[8] + " " + board[7]);
        System.out.println(board[PLAYER2_STORE] + "             " + board[PLAYER1_STORE]);
        System.out.println("  " + board[0] + " " + board[1] + " " + board[2] + " " + 
                          board[3] + " " + board[4] + " " + board[5]);
    }
    
    static class GameState {
        int[] board;
        boolean isPlayerTurn;
        
        GameState(int[] board, boolean isPlayerTurn) {
            this.board = board;
            this.isPlayerTurn = isPlayerTurn;
        }
        
        boolean isGameOver() {
            return isGameSideEmpty(board, 0, 6) || isGameSideEmpty(board, 7, 13);
        }
    }
    
    static class Move {
        int pit;
        int score;
        boolean extraTurn;
        
        Move(int pit, int score, boolean extraTurn) {
            this.pit = pit;
            this.score = score;
            this.extraTurn = extraTurn;
        }
    }
}