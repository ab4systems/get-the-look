package com.shopstyle.bo;

//JdbcTestOracle.java
import java.sql.Connection ;
import java.sql.DriverManager ;
import java.sql.PreparedStatement;
import java.sql.ResultSet ;
import java.sql.SQLException;
import java.sql.Statement ;
import java.sql.Types;
import java.util.ArrayList;

public class DatabaseBuilder {
    
    Connection connection;
    
    void establishConnection() {
        try 
        {
            Class.forName("oracle.jdbc.driver.OracleDriver");
        }
        catch (ClassNotFoundException e) {
            System.err.println (e) ;
            System.exit (-1) ;
        }
        try {
            // open connection to database
            connection = DriverManager.getConnection(
             "jdbc:oracle:thin:@localhost:1521:XE",
             "HR",  // ## fill in User here
             "parola" // ## fill in Password here
            );
       }
       catch (SQLException e)
        {
            System.err.println (e) ;
            System.exit (-1) ;
        }
    }
    void closeConnection() {
        try {
            connection.close();
        }
        catch(SQLException e) {
            e.printStackTrace(); ;
            System.exit (-1) ;
        }
    }
    void insertTableRetailers(ArrayList<Retailer> retailers)
    {
        try 
        {
            establishConnection();
            for (int i=0; i<retailers.size(); i++)
            {
                // build query
                String query = "insert into retailers values(?,?)";               
                PreparedStatement ps = connection.prepareStatement(query);
                ps.setLong(1,retailers.get(i).getId());
                ps.setString(2,retailers.get(i).getName());
                
                // execute query
                ps.execute();
                ps.close();
            }
            closeConnection();
        }
        catch (SQLException e)
        {
            e.printStackTrace(); 
            System.exit (-1) ;
        }
    }
    void insertTableColors(ArrayList<Color> colors)
    {
        try 
        {
            establishConnection();
            for (int i=0; i<colors.size(); i++)
            {
                // build query
                String query = "insert into colors values(?,?)";               
                PreparedStatement ps = connection.prepareStatement(query);
                ps.setLong(1,colors.get(i).getId());
                ps.setString(2,colors.get(i).getName());
                
                // execute query
                ps.execute();
                ps.close();
            }
            closeConnection();
        }
        catch (SQLException e)
        {
            e.printStackTrace(); 
            System.exit (-1) ;
        }
    }
    void insertTableBrands(ArrayList<Brand> brands)
    {
        try 
        {
            establishConnection();
            for (int i=0; i<brands.size(); i++)
            {
                // build query
                String query = "insert into brands values(?,?)";
                PreparedStatement ps = connection.prepareStatement(query);
                ps.setLong(1,brands.get(i).getId());
                ps.setString(2,brands.get(i).getName());

                // execute query
                ps.execute();
                ps.close();
            }
            closeConnection();
        }
        catch (SQLException e)
        {
            e.printStackTrace(); 
            System.exit (-1) ;
        }
    }
    void insertTableCategories (ArrayList<Category> categories) {
      try {
        Class.forName("oracle.jdbc.driver.OracleDriver");
       }
        catch (ClassNotFoundException e) {
            System.err.println (e) ;
            System.exit (-1) ;
        }
       try {
            // open connection to database
            establishConnection();
           for (int i=0; i<categories.size(); i++)
           {
                // build query
                String query = "insert into categories values(?,?,?)";
                PreparedStatement ps = connection.prepareStatement(query);
                String id = categories.get(i).getId();
                String parentId = categories.get(i).getParentId(); 
                ps.setString(1,categories.get(i).getId());
                ps.setString(2,categories.get(i).getName());
                ps.setString(3,categories.get(i).getParentId());
                
                // execute query
                ps.execute();
                ps.close();
            }
            closeConnection();
        }
        catch (SQLException e)
        {
            e.printStackTrace(); 
            System.exit (-1) ;
        }
    }
    void insertTableProducts (ArrayList<Product> products) {
      try {
        Class.forName("oracle.jdbc.driver.OracleDriver");
       }
        catch (ClassNotFoundException e) {
            System.err.println (e) ;
            System.exit (-1) ;
        }
       try {
            // open connection to database
            establishConnection();
            int k = 8112; 
            for (int i=0; i<products.size(); i++)
            {
                if (products.get(i).getColors() != null)
                    for (Color color: products.get(i).getColors())
                        for (Category category: products.get(i).getCategories())
                        {
                            // build query
                            String query = "insert into products values(?,?,?,?,?,?,?,?,?,?,?,?)";
                            PreparedStatement ps = connection.prepareStatement(query); 
                            ps.setInt(1,k);
                            ps.setLong(2,products.get(i).getId());
                            ps.setString(3,products.get(i).getName());
                            if (products.get(i).getRetailer() != null)
                                ps.setLong(4,products.get(i).getRetailer().getId());
                            else ps.setNull(4, Types.NUMERIC);
                            if (products.get(i).getBrand() != null)
                                ps.setLong(5,products.get(i).getBrand().getId());
                            else ps.setNull(5,Types.NUMERIC);
                            ps.setLong(6,color.getId());
                            ps.setString(7,category.getId());
                            ps.setString(8,products.get(i).getClickUrl());
                            ps.setString(9, products.get(i).getImage().getId());
                            ps.setInt(10,products.get(i).getImage().getWidth());
                            ps.setInt(11,products.get(i).getImage().getHeight());
                            ps.setString(12,products.get(i).getImage().getUrl());

                            // execute query
                            ps.execute();
                            ps.close();
                            k++;
                        }
                else 
                    for (Category category: products.get(i).getCategories())
                    {
                        // build query
                        String query = "insert into products values(?,?,?,?,?,?,?,?,?,?,?,?)";
                        PreparedStatement ps = connection.prepareStatement(query); 
                        ps.setInt(1,k);
                        ps.setLong(2,products.get(i).getId());
                        ps.setString(3,products.get(i).getName());
                        if (products.get(i).getRetailer() != null)
                            ps.setLong(4,products.get(i).getRetailer().getId());
                        else ps.setNull(4, Types.NUMERIC);
                        if (products.get(i).getBrand() != null)
                            ps.setLong(5,products.get(i).getBrand().getId());
                        else ps.setNull(5,Types.NUMERIC);
                        ps.setNull(6,Types.NUMERIC);
                        ps.setString(7,category.getId());
                        ps.setString(8,products.get(i).getClickUrl());
                        ps.setString(9, products.get(i).getImage().getId());
                        ps.setInt(10,products.get(i).getImage().getWidth());
                        ps.setInt(11,products.get(i).getImage().getHeight());
                        ps.setString(12,products.get(i).getImage().getUrl());

                        // execute query
                        ps.execute();
                        ps.close();
                        k++;
                    }
            }
            closeConnection();
        }
        catch (SQLException e)
        {
            e.printStackTrace(); 
            System.exit (-1) ;
        }
    }
}