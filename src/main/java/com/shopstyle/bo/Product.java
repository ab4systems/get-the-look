package com.shopstyle.bo;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.Arrays;

@JsonIgnoreProperties(ignoreUnknown = true)
public class Product
{
    private long id;
    private String name;
    private String priceLabel;
    private Retailer retailer;
    private Brand brand;
    private String clickUrl;
    private Image image;
    private Color[] colors;
    private Category[] categories;

    public long getId()
    {
        return id;
    }

    public void setId(long id)
    {
        this.id = id;
    }

    public String getName()
    {
        return name;
    }

    public void setName(String name)
    {
        this.name = name;
    }

    public String getPriceLabel()
    {
        return priceLabel;
    }

    public void setPriceLabel(String priceLabel)
    {
        this.priceLabel = priceLabel;
    }

    public Retailer getRetailer()
    {
        return retailer;
    }

    public void setRetailer(Retailer retailer)
    {
        this.retailer = retailer;
    }

    public Brand getBrand()
    {
        return brand;
    }

    public void setBrand(Brand brand)
    {
        this.brand = brand;
    }

    public String getClickUrl()
    {
        return clickUrl;
    }

    public void setClickUrl(String clickUrl)
    {
        this.clickUrl = clickUrl;
    }

    public Image getImage()
    {
        return image;
    }

    public void setImage(Image image)
    {
        this.image = image;
    }

    public Color[] getColors()
    {
        return colors;
    }

    public void setColors(Color[] colors)
    {
        this.colors = colors;
    }

    public Category[] getCategories()
    {
        return categories;
    }

    public void setCategories(Category[] categories)
    {
        this.categories = categories;
    }
    @Override
    public String toString() {
        return "Product{id=" + id + ", name=" + name +  
                ", brand=" + brand.toString() + ", retailer=" + retailer.toString() +
                ", priceLabel=" + priceLabel + ", inStock=" +
                ", categories=" + Arrays.toString(categories) + ", colors=" +
                Arrays.toString(colors) + ", clickUrl=" + clickUrl + image.toString() + "}";
    }
}
