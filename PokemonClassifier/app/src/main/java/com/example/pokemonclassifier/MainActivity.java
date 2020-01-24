package com.example.pokemonclassifier;

import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Size;
import android.view.TextureView;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    private Size previewSize = new Size(1080, 1440);

    private CameraPreview mCameraPreview;
    private PokemonClassifier mPokemonClassifier;
    private TextView mTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[] {Manifest.permission.CAMERA}, 34);
            while (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {}
        }

        mTextView = (TextView) findViewById(R.id.resultText);
        mPokemonClassifier = new PokemonClassifier(this, mTextView);

        mCameraPreview = (CameraPreview) findViewById(R.id.cameraPreview);
        mCameraPreview.setClassifier(mPokemonClassifier);

    }

    @Override
    protected void onResume() {
        super.onResume();
        mCameraPreview.onResume(this);
    }
    @Override
    protected void onPause() {
        mCameraPreview.onPause();
        super.onPause();
    }
    public void takePic(View view) {
        mCameraPreview.takePicture();
    }
}

