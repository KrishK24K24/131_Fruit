package com.example.a131fruits;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.a131fruits.ml.ConvertedModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.schema.Model;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    TextView result;
    ImageView imageView;
    Button picture;
    int imageSize = 100;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);

        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    //Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
    }

    public void classifyImage(Bitmap image) {
        try {
            ConvertedModel model = ConvertedModel.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 100, 100, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4*imageSize*imageSize*3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int [] intValues = new int[imageSize*imageSize];
            image.getPixels(intValues,0,image.getWidth(),0,0,image.getWidth(),image.getHeight());
            int pixel = 0;
            for(int i = 0; i < imageSize; i++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF)*(1.f/255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF)*(1.f/255.f));
                    byteBuffer.putFloat((val & 0xFF)*(1.f/255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ConvertedModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            for(int i = 0; i < confidences.length; i++){
                if(confidences[i] > maxConfidence){
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }


            String[] classes = {"Apple Braeburn","Apple Crimson Snow","Apple Golden ","Apple Golden","Apple Golden","Apple Granny Smith","Apple Pink Lady","Apple Red ","Apple Red ","Apple Red ","Apple Red Delicious","Apple Red Yellow ","Apple Red Yellow ","Apricot","Avocado","Avocado ripe","Banana","Banana Lady Finger","Banana Red","Beetroot","Blueberry","Cactus fruit","Cantaloupe ","Cantaloupe ","Carambula","Cauliflower","Cherry ","Cherry ","Cherry Rainier","Cherry Wax Black","Cherry Wax Red","Cherry Wax Yellow","Chestnut","Clementine","Cocos","Corn","Corn Husk","Cucumber Ripe","Cucumber Ripe ","Dates","Eggplant","Fig","Ginger Root","Granadilla","Grape Blue","Grape Pink","Grape White","Grape White ","Grape White ","Grape White ","Grapefruit Pink","Grapefruit White","Guava","Hazelnut","Huckleberry","Kaki","Kiwi","Kohlrabi","Kumquats","Lemon","Lemon Meyer","Limes","Lychee","Mandarine","Mango","Mango Red","Mangostan","Maracuja","Melon Piel de Sapo","Mulberry","Nectarine","Nectarine Flat","Nut Forest","Nut Pecan","Onion Red","Onion Red Peeled","Onion White","Orange","Papaya","Passion Fruit","Peach","Peach ","Peach Flat","Pear","Pear ","Pear Abate","Pear Forelle","Pear Kaiser","Pear Monster","Pear Red","Pear Stone","Pear Williams","Pepino","Pepper Green","Pepper Orange","Pepper Red","Pepper Yellow","Physalis","Physalis with Husk","Pineapple","Pineapple Mini","Pitahaya Red","Plum","Plum ","Plum ","Pomegranate","Pomelo Sweetie","Potato Red","Potato Red Washed","Potato Sweet","Potato White","Quince","Rambutan","Raspberry","Redcurrant","Salak","Strawberry","Strawberry Wedge","Tamarillo","Tangelo","Tomato ","Tomato ","Tomato ","Tomato ","Tomato Cherry Red","Tomato Heart","Tomato Maroon","Tomato Yellow","Tomato not Ripened","Walnut","Watermelon"};

            result.setText(classes[maxPos]);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == 1 && resultCode == RESULT_OK) {
            Bitmap image = (Bitmap) data.getExtras().get("data");
            int dimension = Math.min(image.getWidth(),image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image,dimension,dimension);
            imageView.setImageBitmap(image);

            image = Bitmap.createScaledBitmap(image,imageSize,imageSize,false);
            classifyImage(image);

        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}