package com.example.pokemonclassifier;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.widget.TextView;

import com.google.firebase.ml.common.FirebaseMLException;
import com.google.firebase.ml.custom.FirebaseCustomLocalModel;
import com.google.firebase.ml.custom.FirebaseModelDataType;
import com.google.firebase.ml.custom.FirebaseModelInputOutputOptions;
import com.google.firebase.ml.custom.FirebaseModelInputs;
import com.google.firebase.ml.custom.FirebaseModelInterpreter;
import com.google.firebase.ml.custom.FirebaseModelInterpreterOptions;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class PokemonClassifier {

    private int inputSize = 224;
    private static final float IMAGE_RESCALE = 255.0f;
    private int num_classes = 150;

    private String modelFilename = "model.tflite";
    private String labelFilename = "labelmap.txt";

    private TextView resultView;

    private FirebaseModelInputOutputOptions inputOutputOptions;
    private FirebaseModelInterpreter interpreter;

    private ArrayList<String> labels = new ArrayList<>();

    public PokemonClassifier(Context context, TextView res) {

        FirebaseCustomLocalModel localModel = new FirebaseCustomLocalModel.Builder()
                .setAssetFilePath(modelFilename)
                .build();

        try {
            FirebaseModelInterpreterOptions options =
                    new FirebaseModelInterpreterOptions.Builder(localModel).build();
            interpreter = FirebaseModelInterpreter.getInstance(options);

            inputOutputOptions =
                    new FirebaseModelInputOutputOptions.Builder()
                            .setInputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, inputSize, inputSize, 3})
                            .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, num_classes})
                            .build();
        } catch (FirebaseMLException e) {
            e.printStackTrace();
        }

        try {
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(context.getAssets().open(labelFilename)));
            String line;
            // reads all lables in order to convert model output into Pokemon's name
            while ((line = reader.readLine()) != null) {
                labels.add(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // TextureView for displaying result
        resultView = res;
    }

    public void classifyImage(Bitmap frame) {

        int[] intValues = new int[inputSize * inputSize];
        frame = Bitmap.createScaledBitmap(frame, inputSize, inputSize, true);
        frame.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize);

        float[][][][] imgData = new float[1][224][224][3];
        for (int i = 0; i < inputSize; ++ i) {
            for (int j = 0; j < inputSize; ++j) {
                int pixelValue = intValues[i * inputSize + j];
                // 0xRRGGBB
                imgData[0][i][j][0] = (float) Color.red(pixelValue) / IMAGE_RESCALE;
                imgData[0][i][j][1] = (float) Color.green(pixelValue) / IMAGE_RESCALE;
                imgData[0][i][j][2] = (float) Color.blue(pixelValue) / IMAGE_RESCALE;
            }
        }

        FirebaseModelInputs inputs = null;
        try {
            inputs = new FirebaseModelInputs.Builder()
                    .add(imgData)
                    .build();
        } catch (FirebaseMLException e) {
            e.printStackTrace();
        }
        interpreter.run(inputs, inputOutputOptions)
                .addOnSuccessListener(
                        result -> {
                            float[][] outputs = result.getOutput(0);
                            int prediction = getPrediction(outputs[0]);
                            float probability = outputs[0][prediction];
                            String message = labels.get(prediction) + "\n" + String.format("%.2f", probability * 100) + "%";

                            // show result on the TextureView
                            resultView.setTextSize(20);
                            resultView.setTextColor(Color.WHITE);
                            resultView.setText(message);
                        });
    }

    private int getPrediction(float[] output) {
        int maxIndex = 0;
        float maxValue = 0f;

        for (int i = 0; i < output.length; i ++) {
            if (output[i] > maxValue) {
                maxValue = output[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

}
