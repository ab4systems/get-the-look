create table colors (id number(3) primary key,
                    name varchar2(20));
                    
create table retailers (id number(5) primary key,
                        name varchar2(50));
                        
create table brands (id number(5) primary key,
                    name varchar2(50));
                  
select * from colors;

drop table retailers;

select count(*) from brands;

select * from retailers;

drop table brands;

create table categories (id varchar2(50) primary key,
                         name varchar2(50),
                         parentId varchar2(50));
            
drop table categories;

select * from categories;

select count(*) from categories;
                    
select id, parentId
from categories c
where exists (select 1 from categories d where c.parentId = d.id);

select id
from categories
where parentId = 'kids-and-baby';

insert into categories
values ('clothes-shoes-and-jewelry', 'Clothes Shoes and Jewelry', null);

create table products (id number(20) primary key,
                       productId number(20),
                       name varchar2(200),
                       retailerId number(5),
                       brandId number(5),
                       colorId number(3),
                       categoryId varchar2(50),
                       clickUrl varchar2(100),
                       imageId varchar2(50),
                       width number(4),
                       height number(4),
                       imageUrl varchar2(100));          

alter table categories
add constraint fk_categories foreign key (parentId) references categories (id) on delete cascade;

drop table products;

select count(*) from products;

alter table products
add constraint fk_prod_retailer foreign key (retailerId) references retailers (id) on delete cascade;

alter table products
add constraint fk_prod_brand foreign key (brandId) references brands (id) on delete cascade;

select * from products
where not exists (select 1 from brands where brandid = id) and brandId is not null;--exista branduri care nu sunt in tabelul de branduri

alter table products
add constraint fk_prod_color foreign key (colorId) references colors (id) on delete cascade;

alter table products
add constraint fk_prod_category foreign key (categoryId) references categories (id) on delete cascade;

delete from categories
where id = 'living';

delete from categories
where id = 'mens-grooming' or id = 'womens-beauty';

delete from categories 
where id = 'baby-gear' or id = 'nursery-kids-and-baby' or id = 'kids-and-baby-bath' or id = 'kids-bedroom';