package com.shopstyle.bo;

import java.util.ArrayList;
import java.util.Iterator;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.net.URL;
import java.util.Arrays;

public class ReadFromApi {
    static ObjectMapper objectMapper = new ObjectMapper();
    static String colorsURL = "http://api.shopstyle.com/api/v2/colors?pid=uid5296-37250464-70";
    static String brandsURL = "http://api.shopstyle.com/api/v2/brands?pid=uid5296-37250464-70";
    static String retailersURL = "http://api.shopstyle.com/api/v2/retailers?pid=uid5296-37250464-70";
    static String categoriesURL = "http://api.shopstyle.com/api/v2/categories?pid=uid5296-37250464-70";
    static String productsURL = "http://api.shopstyle.com/api/v2/products?pid=uid5296-37250464-70&";
    static ArrayList<Color> colors = new ArrayList<>();
    static ArrayList<Brand> brands = new ArrayList<>();
    static ArrayList<Retailer> retailers = new ArrayList<>();
    static ArrayList<Category> categories = new ArrayList<>();
    static ArrayList<Product> products = new ArrayList<>();
    
    static void readColors() throws Exception{
        //read JSON like DOM Parser
        JsonNode rootNode = objectMapper.readTree(new URL(colorsURL));
        JsonNode colorsNode = rootNode.path("colors");
        Iterator<JsonNode> elements = colorsNode.elements();
        while(elements.hasNext()){
            JsonNode color = elements.next();
            colors.add(objectMapper.treeToValue(color, Color.class));
        }
        //for (int i=0;i<colors.size();i++)
        //    System.out.println(colors.get(i).toString());
    }
    
    static void readRetailers() throws Exception {
        //read JSON like DOM Parser
        JsonNode rootNode = objectMapper.readTree(new URL(retailersURL));
        JsonNode retailersNode = rootNode.path("retailers");
        Iterator<JsonNode> elements = retailersNode.elements();
        while(elements.hasNext()){
            JsonNode retailer = elements.next();
            retailers.add(objectMapper.treeToValue(retailer, Retailer.class));
        }
        for (int i=0;i<retailers.size();i++)
            System.out.println(retailers.get(i).toString());
    }
    static void readBrands() throws Exception {
        //read JSON like DOM Parser
        JsonNode rootNode = objectMapper.readTree(new URL(brandsURL));
        JsonNode brandsNode = rootNode.path("brands");
        Iterator<JsonNode> elements = brandsNode.elements();
        while(elements.hasNext()){
            JsonNode brand = elements.next();
            brands.add(objectMapper.treeToValue(brand, Brand.class));
        }
        for (int i=0;i<brands.size();i++)
            System.out.println(brands.get(i).toString());
    }
    static  void readCategories() throws Exception {
        //read JSON like DOM Parser
        JsonNode rootNode = objectMapper.readTree(new URL(categoriesURL));
        JsonNode categoriesNode = rootNode.path("categories");
        Iterator<JsonNode> elements = categoriesNode.elements();
        while(elements.hasNext()){
            JsonNode category = elements.next();
            categories.add(objectMapper.treeToValue(category, Category.class));
        }
        for (int i=0;i<categories.size();i++)
            System.out.println(categories.get(i).toString());
    }
    static  void readProducts() throws Exception {
        //read JSON like DOM Parser
        Integer k = 5000;
        while (k < 10000)
        {
            String offset = k.toString();
            JsonNode rootNode = objectMapper.readTree(new URL(productsURL + "&offset=" + offset + "&limit=50"));
            JsonNode productsNode = rootNode.path("products");
            Iterator<JsonNode> elements = productsNode.elements();
            while(elements.hasNext()){
                JsonNode product = elements.next();
                Product p = new Product();
                p.setId(product.get("id").asLong());
                p.setName(product.get("name").asText());
                p.setBrand(objectMapper.convertValue(product.get("brand"),Brand.class));
                p.setClickUrl(product.get("clickUrl").asText());
                p.setPriceLabel(product.get("priceLabel").asText());
                p.setRetailer(objectMapper.convertValue(product.get("retailer"),Retailer.class));
                JsonNode categoriesNode = product.path("categories");
                p.setCategories(objectMapper.treeToValue(categoriesNode, Category[].class));
                JsonNode colorsNode = product.path("colors");
                Iterator<JsonNode> colors = colorsNode.elements();
                ArrayList<Color> c = new ArrayList<>();
                while (colors.hasNext())
                {
                    Color[] aux = objectMapper.treeToValue(colors.next().get("canonicalColors"),Color[].class);
                    c.addAll(new ArrayList<>(Arrays.asList(aux)));
                }
                Color[] colorArray = c.toArray(new Color[0]);
                p.setColors(colorArray);
                JsonNode imageNode = product.get("image");
                Image image = new Image();
                image.setId(imageNode.get("id").asText());
                imageNode = imageNode.get("sizes").get("Best");
                image.setHeight(imageNode.get("actualHeight").asInt());
                image.setWidth(imageNode.get("actualWidth").asInt());
                image.setUrl(imageNode.get("url").asText());
                p.setImage(image);
                products.add(p);
                k++;
            }
        }
        //for (int i=0;i<55;i++)
            //System.out.println(String.valueOf(i) + " " + products.get(i).toString());
    }
}
