package com.shopstyle.bo;

public class Application {

    static ReadFromApi readFromApi = new ReadFromApi();
    static DatabaseBuilder databaseBuilder = new DatabaseBuilder();
    public static void main(String[] args) throws Exception {
        //readFromApi.readColors();
        // readFromApi.readBrands();
        // readFromApi.readRetailers();
        // readFromApi.readCategories();
        readFromApi.readProducts();
        // databaseBuilder.insertTableColors(readFromApi.colors);
        // databaseBuilder.insertTableRetailers(readFromApi.retailers);
        // databaseBuilder.insertTableBrands(readFromApi.brands);
        // databaseBuilder.insertTableCategories(readFromApi.categories);
        databaseBuilder.insertTableProducts(readFromApi.products);
               
    }
    
	

}