package com.github.nidorx.jia.mlp;

import static org.junit.Assert.assertEquals;
import org.junit.Test;

/**
 * Testado usando o {@link TransferViewer}
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public class TransferTest {

    @Test
    public void testIDENTITY() {
        validate(
                Transfer.IDENTITY,
                new double[][]{
                    new double[]{-1.0, -1.0, 1.0},
                    new double[]{0.0, 0.0, 1.0},
                    new double[]{1.0, 1.0, 1.0},}
        );
    }

    @Test
    public void testBENT_IDENTITY() {
        validate(
                Transfer.BENT_IDENTITY,
                new double[][]{
                    new double[]{-0.9955645161290323, -0.7900241725314401, 0.6472333137419932},
                    new double[]{-0.5088709677419354, -0.4478563374044963, 0.7732363203095703},
                    new double[]{0.0, 0.0, 1.0},
                    new double[]{0.4919354838709680, 0.5491608692159294, 1.2207075883515364},
                    new double[]{0.9991935483870971, 1.2060152633793568, 1.3534107424805275}
                }
        );
    }

    @Test
    public void testSIGMOID() {
        validate(
                Transfer.SIGMOID,
                new double[][]{
                    new double[]{-4.0000000000000000, 0.0179862099620916, 0.0176627062132911},
                    new double[]{-1.6380952380952385, 0.1627244105118250, 0.1362451767354040},
                    new double[]{-0.1904761904761907, 0.4525244048739520, 0.2477460678674275},
                    new double[]{0.9904761904761905, 0.7291819686232285, 0.1974756252579815},
                    new double[]{2.9904761904761905, 0.9521420136794535, 0.0455675994658889}
                }
        );
    }

    @Test
    public void testTANH() {
        validate(
                Transfer.TANH,
                new double[][]{
                    new double[]{-2.0032258064516126, -0.9642547786399196, 0.0702127218700797},
                    new double[]{-0.5225806451612902, -0.4796893043034112, 0.7698981713369094},
                    new double[]{0.0870967741935482, 0.0868772061145759, 0.9924523510577254},
                    new double[]{0.9870967741935486, 0.7561216575421131, 0.4282800389957674},
                    new double[]{1.9451612903225808, 0.9599412472487260, 0.0785128018305603}
                }
        );
    }

    @Test
    public void testARCTAN() {
        validate(
                Transfer.ARCTAN,
                new double[][]{
                    new double[]{-2.5258064516129030, -1.1938180497075224, 0.1355068959050409},
                    new double[]{-1.0161290322580645, -0.7933979927373349, 0.4920005119672342},
                    new double[]{0.4790322580645166, 0.4467331519876074, 0.8133573419041954},
                    new double[]{1.0306451612903231, 0.8004883602117349, 0.4849120939145044},
                    new double[]{2.0177419354838717, 1.1106720856613290, 0.1971887774757476}
                }
        );
    }

    @Test
    public void testSOFTSIGN() {
        validate(
                Transfer.SOFTSIGN,
                new double[][]{
                    new double[]{-4.0063897763578270, -0.8002552648372686, 0.3085544545447806},
                    new double[]{-1.4952076677316288, -0.5992317541613316, 0.3910003903843651},
                    new double[]{0.1150159744408947, 0.1031518624641835, 0.8217304773148925},
                    new double[]{1.6485623003194894, 0.6224366706875755, 0.3798958002238775},
                    new double[]{3.5271565495207664, 0.7791107974594212, 0.3159322854228369}
                }
        );
    }

    @Test
    public void testISRU() {
        validate(
                Transfer.ISRU,
                new double[][]{
                    new double[]{-2.4967741935483870, -0.9272277730233085, 0.0512178676846116},
                    new double[]{-1.0016129032258063, -0.7074863171655450, 0.3524150143908375},
                    new double[]{-0.2903225806451610, -0.2787080801619360, 0.8847207890853238},
                    new double[]{0.2177419354838710, 0.2125875766332337, 0.9306521967875450},
                    new double[]{3.0193548387096776, 0.9476176302521796, 0.0309141230729962}
                }
        );
    }

    @Test
    public void testISRLU() {
        validate(
                Transfer.ISRLU,
                new double[][]{
                    new double[]{-2.8466453674121404, -0.9423015539324360, 0.0362718429558734},
                    new double[]{-1.8546325878594252, -0.8800482538598040, 0.1068429319116863},
                    new double[]{-0.6613418530351440, -0.5509902736489886, 0.5783007744298999},
                    new double[]{0.0862619808306704, 0.0862619808306704, 1.0000000000000000},
                    new double[]{0.7907348242811496, 0.7907348242811496, 1.0000000000000000}
                }
        );
    }
    
    @Test
    public void testELU() {
        validate(
                Transfer.ELU,
                new double[][]{
                    new double[]{-3.1054313099041533, -0.9551948105308230, 0.0448051894691770},
                    new double[]{-1.5910543130990416, -0.7962892764098700, 0.2037107235901300},
                    new double[]{-0.2683706070287535, -0.2353756465737723, 0.7646243534262277},
                    new double[]{0.0575079872204478, 0.0575079872204478, 1.0000000000000000},
                    new double[]{0.9009584664536741, 0.9009584664536741, 1.0000000000000000}
                }
        );
    }

    @Test
    public void testRELU() {
        validate(
                Transfer.RELU,
                new double[][]{
                    new double[]{-0.6201923076923075, 0.0000000000000000, 0.0000000000000000},
                    new double[]{0.0865384615384617, 0.0865384615384617, 1.0000000000000000}
                }
        );
    }

    @Test
    public void testLRELU() {
        validate(
                Transfer.LRELU,
                new double[][]{
                    new double[]{-3.7380191693290734, -0.7476038338658147, 0.0100000000000000},
                    new double[]{-2.3003194888178910, -0.4600638977635783, 0.0100000000000000},
                    new double[]{-0.3258785942492013, -0.0651757188498403, 0.0100000000000000},
                    new double[]{0.2683706070287544, 0.2683706070287544, 1.0000000000000000},
                    new double[]{1.5910543130990416, 1.5910543130990416, 1.0000000000000000}
                }
        );
    }

    @Test
    public void testSOFTPLUS() {
        validate(
                Transfer.SOFTPLUS,
                new double[][]{
                    new double[]{-2.9142857142857140, 0.0528227481041030, 0.0514518704633653},
                    new double[]{-1.7142857142857144, 0.1655926660515749, 0.1526086648426310},
                    new double[]{-0.0761904761904759, 0.6557773906055048, 0.4809615898743669},
                    new double[]{0.1523809523809527, 0.7722373472630806, 0.5380216947161702},
                    new double[]{2.1142857142857140, 2.2282565924228246, 0.8922839404746193}
                }
        );
    }

    @Test
    public void testGAUSSIAN() {
        validate(
                Transfer.GAUSSIAN,
                new double[][]{
                    new double[]{-2.0702875399361020, 0.0137586096010929, 0.0569685560479757},
                    new double[]{-1.1501597444089455, 0.2663704053118147, 0.6127370345830880},
                    new double[]{-0.3642172523961662, 0.8757678742062136, 0.6379395377604369},
                    new double[]{0.1533546325878596, 0.9767567412692867, -0.2995803423701331},
                    new double[]{1.4760383386581477, 0.1131898052170546, -0.3341449840912413}
                }
        );
    }

    private void validate(final Transfer transfer, double[][] values) {
        for (double[] value : values) {
            double input = value[0];
            double expectedActvation = value[1];
            double expectedDerivative = value[2];

            assertEquals(expectedActvation, transfer.activation(input), 0.0000000000001);
            assertEquals(expectedDerivative, transfer.derivative(expectedActvation, input), 0.0000000000001);
        }
    }

}
