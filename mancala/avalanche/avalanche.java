
import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);

        // Take in 12 numbers CCWISE
        System.out.println("The following solves the GamePigeon mancala game on avalanche mode. Input the number of stones in each space starting in the top left counterclockwise. Do not include your frontmost (score) space. You will receive a string of numbers between 0 and 5. 0 is the top left, 5 is the bottom left.");

        int[] game = new int[13];

        game[0] = input.nextInt();
        game[1] = input.nextInt();
        game[2] = input.nextInt();
        game[3] = input.nextInt();
        game[4] = input.nextInt();
        game[5] = input.nextInt();

        // Storage zone
        game[6] = 0;

        game[7] = input.nextInt();
        game[8] = input.nextInt();
        game[9] = input.nextInt();
        game[10] = input.nextInt();
        game[11] = input.nextInt();
        game[12] = input.nextInt();

       System.out.println("Calculating...");

        Object[] vals = bestMove(game);
        System.out.println((String)vals[1] + " TOTAL: " + (int)vals[0]);

    }
private static int cascade(int selection, int[] game) {
        int store = game[selection];
        game[selection] = 0;
        int i = selection + 1;
        for (;; i++) {
            if (i >= 13) {
                i -= 13;
            }
            game[i]++;
            store--;
            if(store == 0)
                break;
        }
        if (game[i] > 1 && i != 6) {
            i = cascade(i, game);
        }
        return i;
    }
    public static Object[] bestMove(int[] game) {
        Object[] maxReturns = {0, ""};
        for (int i = 0; i < 6; i++) {
            int[] gameCopy = game.clone();
            int endOfMove = i;
            if (gameCopy[i] != 0) {
                endOfMove = cascade(i, gameCopy);
            }
            Object[] returns = {gameCopy[6], i + 1 + ""};
            if (endOfMove == 6) {
                Object[] n = bestMove(gameCopy);
                returns[0] = n[0];
                returns[1] = (String)returns[1] + " " + (String)n[1];
            }
            if ((int) returns[0] > (int) maxReturns[0]) {
                maxReturns = returns;
            }
        }
        return maxReturns;
    }

}
