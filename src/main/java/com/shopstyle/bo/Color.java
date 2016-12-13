package com.shopstyle.bo;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * Normalized color of a product
 */
 @JsonIgnoreProperties(ignoreUnknown = true)
public class Color implements SearchFilter
{
    private long id;
    private String name;

    @JsonCreator
    public Color(@JsonProperty("id") long id, @JsonProperty("name") String name)
    {
        this.id = id;
        this.name = name;
    }
    
    @Override
    public String getFilterId()
    {
        return "c" + getId();
    }
    
    /**
     * Returns the unique identifier of this color
     */
    public long getId()
    {
        return id;
    }

    /**
     * Returns the display name of this color
     */
    public String getName()
    {
        return name;
    }

   public void setId (long i)
   {
        id = i;
   }
   public void setName (String n)
   {
        name = n;
   }
   public String toString() 
   {
	return "Color{" + "id=" + id + ", name=" + name + "}";
   }
}
