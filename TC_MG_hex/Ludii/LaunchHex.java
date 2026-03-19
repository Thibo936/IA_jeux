import app.DesktopApp;
import app.StartDesktopApp;
import java.util.Arrays;
import javax.swing.SwingUtilities;

public class LaunchHex {

    private static final String HEX_GAME = "/lud/board/space/connection/Hex.lud";

    public static void main(String[] args) {
        StartDesktopApp.main(new String[0]);

        SwingUtilities.invokeLater(() -> {
            final DesktopApp desktopApp = StartDesktopApp.desktopApp();
            if (desktopApp == null) {
                System.err.println("LaunchHex: desktop app not available.");
                return;
            }

            desktopApp.loadGameFromName(
                HEX_GAME,
                Arrays.asList("Board Size/11x11", "Swap Rules/Off", "End Rules/Standard"),
                false
            );
        });
    }
}