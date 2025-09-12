from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.flipkart.com/fk-sasalele-sale-tv-and-appliances-may25-at-store?param=3783&fm=neo%2Fmerchandising&iid=M_1bc1c2ec-322e-4acd-aecd-0067cf757dd4_1_X1NCR146KC29_MC.YX88A89LFA7C&otracker=hp_rich_navigation_6_1.navigationCard.RICH_NAVIGATION_TVs%2B%26%2BAppliances_YX88A89LFA7C&otracker1=hp_rich_navigation_PINNED_neo%2Fmerchandising_NA_NAV_EXPANDABLE_navigationCard_cc_6_L0_view-all&cid=YX88A89LFA7C")
docs = loader.load()


print(docs[0].page_content)
